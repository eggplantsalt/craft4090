"""train_craft.py

Scaffold training entrypoint for CRaFT (Constrained Retention Fine-Tuning) on top of
the native OpenVLA training flow.

Current stage (Prompt scaffold):
- Reuse baseline OpenVLA config/loading/dataset pipeline.
- Add CRaFT config surface.
- Execute a one-batch forward dry-run and exit.

Deferred (TODO):
- Two-pass backward for action + retention gradients.
- Conflict-aware gradient projection.
- Primal-dual lambda update and epsilon scheduling.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.craft import CraftConfig
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator

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
    dry_run_batches: int = 1
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


@draccus.wrap()
def train_craft(cfg: TrainCraftConfig) -> None:
    """Run one-batch OpenVLA forward dry-run with CRaFT config loaded."""
    overwatch.info("OpenVLA CRaFT Dry-Run :: Warming Up")

    # Device setup (matches baseline assumptions)
    assert torch.cuda.is_available(), "train_craft assumes at least one GPU is available"
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Build run id for parity/debug readability
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"craft-dryrun+{vla_id}+b{cfg.per_device_batch_size}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
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
    vlm = vlm.to(device_id)
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

    # Minimal one-batch DataLoader path for dry-run
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # TODO(CRaFT): attach anchor-cache loader from `prismatic.craft.anchor_cache`
    # TODO(CRaFT): compute `L_ret` via `prismatic.craft.retention_loss`
    # TODO(CRaFT): run 2-pass backward + gradient surgery + primal-dual lambda update
    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.vla.enable_mixed_precision_training):
            output: CausalLMOutputWithPast = vlm(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(device_id),
                labels=batch["labels"],
            )

        loss_value = float(output.loss.detach().cpu())
        overwatch.info(
            "CRaFT dry-run batch stats :: "
            f"batch_idx={batch_idx}, "
            f"input_ids={tuple(batch['input_ids'].shape)}, "
            f"attention_mask={tuple(batch['attention_mask'].shape)}, "
            f"labels={tuple(batch['labels'].shape)}, "
            f"pixel_values={tuple(batch['pixel_values'].shape)}, "
            f"loss={loss_value:.6f}"
        )
        overwatch.info(f"Action tokenizer begin idx: {action_tokenizer.action_token_begin_idx}")

        if (batch_idx + 1) >= cfg.dry_run_batches:
            break

    overwatch.info("CRaFT dry-run completed; exiting before optimizer/backward integration.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train_craft()
