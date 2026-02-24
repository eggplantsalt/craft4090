"""train_craft.py

CRaFT training entrypoint on top of the native OpenVLA flow.

This script keeps the baseline action loss definition untouched and adds:
- hidden retention loss from offline anchor cache
- conflict-aware gradient projection
- primal-dual lambda update for retention budget control
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import draccus
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.craft import CraftConfig
from prismatic.craft.anchor_cache import build_anchor_dataloader
from prismatic.craft.grad_surgery import compute_dot, merge_grads, project_if_conflict
from prismatic.craft.primal_dual import epsilon_schedule, update_lambda
from prismatic.craft.retention_loss import compute_hidden_retention_loss
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from transformers import LlamaTokenizerFast

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainCraftConfig:
    # fmt: off

    # VLA base config and CRaFT extension config
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )
    craft: CraftConfig = field(default_factory=CraftConfig)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    run_root_dir: Path = Path("runs")

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None
    is_resume: bool = True
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None

    # Run Arguments
    run_id: Optional[str] = None
    run_id_note: Optional[str] = None
    image_aug: bool = False
    seed: int = 7
    dry_run_batches: int = 3
    skip_world_size_check: bool = True

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")

    # Tracking Parameters (kept for parity with baseline script)
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"

    def __post_init__(self) -> None:
        """Lift common optimization parameters from VLA config for baseline parity."""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        if not self.skip_world_size_check:
            assert (
                self.vla.expected_world_size == overwatch.world_size()
            ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


def _cast_pixel_values_for_model(model: Any, pixel_values: Any) -> Any:
    target_dtype = model.llm_backbone.half_precision_dtype if model.enable_mixed_precision_training else torch.float32
    if torch.is_tensor(pixel_values):
        if pixel_values.is_floating_point():
            return pixel_values.to(target_dtype)
        return pixel_values
    if isinstance(pixel_values, dict):
        out: Dict[str, Any] = {}
        for key, value in pixel_values.items():
            if torch.is_tensor(value) and value.is_floating_point():
                out[key] = value.to(target_dtype)
            else:
                out[key] = value
        return out
    return pixel_values


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(val, device) for key, val in value.items()}
    return value


def _clone_grads(params: Sequence[torch.nn.Parameter]) -> list[Optional[torch.Tensor]]:
    grads = []
    for param in params:
        grads.append(param.grad.detach().clone() if param.grad is not None else None)
    return grads


def _backward(loss: torch.Tensor, accelerator: Optional[Any] = None) -> None:
    if accelerator is not None and hasattr(accelerator, "backward"):
        accelerator.backward(loss)
    else:
        loss.backward()


def _append_empty_llama_token_if_needed(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(tokenizer, LlamaTokenizerFast) and not torch.all(input_ids[:, -1] == 29871):
        suffix = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device).repeat(input_ids.shape[0], 1)
        input_ids = torch.cat((input_ids, suffix), dim=1)
        suffix_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat((attention_mask, suffix_mask), dim=1)
    return input_ids, attention_mask


def preprocess_model_inputs_like_train(model: Any, batch_like: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Align batch preprocessing between train batch and anchor batch model inputs."""
    prepared = _to_device(batch_like, device)
    tokenizer = model.llm_backbone.get_tokenizer()

    if "input_ids" in prepared and "attention_mask" in prepared:
        prepared["input_ids"], prepared["attention_mask"] = _append_empty_llama_token_if_needed(
            prepared["input_ids"], prepared["attention_mask"], tokenizer
        )

    if "pixel_values" in prepared:
        prepared["pixel_values"] = _cast_pixel_values_for_model(model, prepared["pixel_values"])

    return prepared


def _grad_cosine(grad_a: Sequence[Optional[torch.Tensor]], grad_b: Sequence[Optional[torch.Tensor]], eps: float = 1e-12) -> float:
    dot = compute_dot(grad_a, grad_b)
    norm_a = torch.sqrt(compute_dot(grad_a, grad_a) + eps)
    norm_b = torch.sqrt(compute_dot(grad_b, grad_b) + eps)
    cos = dot / (norm_a * norm_b + eps)
    return float(cos.detach().cpu().item())


def update_policy_craft(
    model: Any,
    train_batch: Dict[str, Any],
    anchor_batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    craft_cfg: CraftConfig,
    lambda_value: float,
    epsilon_value: float,
    device: torch.device,
    accelerator: Optional[Any] = None,
) -> tuple[Dict[str, float], float]:
    """One CRaFT optimization step with two backward passes and gradient surgery.

    We use two independent forwards (train batch and anchor batch), so `retain_graph=True`
    is not required and memory behavior is more stable.
    """
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]

    processed_train = preprocess_model_inputs_like_train(model, train_batch, device)
    processed_anchor_inputs = preprocess_model_inputs_like_train(model, anchor_batch["model_inputs"], device)
    processed_anchor = {**anchor_batch, "model_inputs": processed_anchor_inputs}
    if processed_anchor.get("target_features") is not None:
        processed_anchor["target_features"] = processed_anchor["target_features"].to(device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.enable_mixed_precision_training):
        output: CausalLMOutputWithPast = model(
            input_ids=processed_train["input_ids"],
            attention_mask=processed_train["attention_mask"],
            pixel_values=processed_train["pixel_values"],
            labels=processed_train["labels"],
        )
        loss_act = output.loss
    _backward(loss_act, accelerator=accelerator)
    grad_act = _clone_grads(params)

    optimizer.zero_grad(set_to_none=True)
    loss_ret, ret_metrics = compute_hidden_retention_loss(model, processed_anchor, craft_cfg)
    _backward(loss_ret, accelerator=accelerator)
    grad_ret = _clone_grads(params)

    grad_dot = compute_dot(grad_act, grad_ret)
    grad_dot_value = float(grad_dot.detach().cpu().item())
    grad_cos_value = _grad_cosine(grad_act, grad_ret, eps=craft_cfg.delta)
    projected = grad_dot_value < 0.0

    grad_act_proj = project_if_conflict(grad_act, grad_ret, craft_cfg.delta)
    merged = merge_grads(grad_act_proj, grad_ret, lambda_value=lambda_value)

    optimizer.zero_grad(set_to_none=True)
    for param, grad in zip(params, merged):
        param.grad = None if grad is None else grad.to(param.device)
    optimizer.step()

    lambda_next = update_lambda(
        lambda_value=lambda_value,
        lambda_lr=craft_cfg.lambda_lr,
        retention_loss=float(loss_ret.detach().cpu().item()),
        epsilon=float(epsilon_value),
        lambda_max=craft_cfg.lambda_max,
    )

    log_metrics = {
        "L_act": float(loss_act.detach().cpu().item()),
        "L_ret": float(loss_ret.detach().cpu().item()),
        "lambda": float(lambda_value),
        "epsilon": float(epsilon_value),
        "grad_dot": grad_dot_value,
        "grad_cos": grad_cos_value,
        "projected": 1.0 if projected else 0.0,
        "hidden_layer": float(ret_metrics["hidden_layer"]),
        "feature_dim": float(ret_metrics["feature_dim"]),
    }
    return log_metrics, lambda_next


@draccus.wrap()
def train_craft(cfg: TrainCraftConfig) -> None:
    """Run CRaFT training loop (supports short dry-run with optimizer updates)."""
    overwatch.info("OpenVLA CRaFT Training :: Warming Up")

    # Device setup (matches baseline assumptions)
    assert torch.cuda.is_available(), "train_craft assumes at least one GPU is available"
    torch.cuda.set_device(device_id := overwatch.local_rank())
    device = torch.device("cuda", device_id)
    torch.cuda.empty_cache()

    # Build run id for parity/debug readability
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"craft+{vla_id}+b{cfg.per_device_batch_size}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"

    # Seed setup (same utility as baseline)
    _ = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Load HF token
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # Load VLA checkpoint (resume) or base VLM
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch
        vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True)
    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    # Preserve baseline stage/freeze semantics
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"
    else:
        raise ValueError("Unsupported VLA freeze configuration")

    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)
    vlm = vlm.to(device)
    vlm.train()

    # Build dataset + collator exactly as baseline VLA train script
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # DataLoader path (same baseline data pipeline)
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    if not cfg.craft.enabled:
        raise ValueError("craft.enabled must be True for train_craft.py")
    if cfg.craft.retention_mode != "hidden":
        raise ValueError(f"train_craft currently integrates hidden retention only; got {cfg.craft.retention_mode}")

    anchor_loader = build_anchor_dataloader(
        cache_dir=cfg.craft.anchor_cache_dir,
        batch_size=cfg.craft.anchor_batch_size,
        num_workers=cfg.craft.anchor_num_workers,
        shuffle=True,
    )
    anchor_iter = iter(anchor_loader)

    optimizer = AdamW([param for param in vlm.parameters() if param.requires_grad], lr=cfg.learning_rate)
    lambda_value = float(cfg.craft.lambda_init)

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) > cfg.dry_run_batches:
            break

        try:
            anchor_batch = next(anchor_iter)
        except StopIteration:
            anchor_iter = iter(anchor_loader)
            anchor_batch = next(anchor_iter)

        epsilon_value = epsilon_schedule(
            step=batch_idx,
            max_steps=cfg.max_steps,
            epsilon_start=cfg.craft.epsilon,
            epsilon_final=cfg.craft.epsilon_final,
            schedule_type=cfg.craft.epsilon_schedule_type,
        )

        metrics, lambda_value = update_policy_craft(
            model=vlm,
            train_batch=batch,
            anchor_batch=anchor_batch,
            optimizer=optimizer,
            craft_cfg=cfg.craft,
            lambda_value=lambda_value,
            epsilon_value=epsilon_value,
            device=device,
            accelerator=None,
        )

        overwatch.info(
            "CRaFT step :: "
            f"step={batch_idx + 1}, "
            f"retention_mode={cfg.craft.retention_mode}, "
            f"L_act={metrics['L_act']:.6f}, "
            f"L_ret={metrics['L_ret']:.6f}, "
            f"lambda={metrics['lambda']:.6f}, "
            f"epsilon={metrics['epsilon']:.6f}, "
            f"grad_dot={metrics['grad_dot']:.6f}, "
            f"grad_cos={metrics['grad_cos']:.6f}, "
            f"projection={bool(metrics['projected'])}"
        )

    overwatch.info("CRaFT dry-run completed with optimizer updates.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train_craft()
