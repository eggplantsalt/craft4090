"""build_anchor_hidden_cache.py

Offline builder for CRaFT hidden-feature anchor cache.

Design constraints:
- Teacher model (theta_0) is loaded only in this offline stage.
- Saved cache stores pooled fixed-length target features, never full token x dim tensors.
- Saved records include `model_inputs` for direct student forward replay without re-tokenizing.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import draccus
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models import load, load_vla
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch
from prismatic.util.data_utils import tree_map
from prismatic.vla.datasets import RLDSDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)


def _parse_dtype(dtype_str: str) -> torch.dtype:
    value = dtype_str.lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _move_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_to_cpu(v) for k, v in value.items()}
    return value


def _cast_floats(value: Any, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype)
        return value
    if isinstance(value, dict):
        return {k: _cast_floats(v, dtype) for k, v in value.items()}
    return value


def _make_prompt(prompt_builder_fn: type[PromptBuilder], instruction: str) -> str:
    builder = prompt_builder_fn("openvla")
    builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    return builder.get_prompt()


def _maybe_append_empty_llama_token(input_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    if isinstance(tokenizer, LlamaTokenizerFast):
        if not torch.all(input_ids[:, -1] == 29871):
            suffix = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat((input_ids, suffix), dim=1)
    return input_ids


class AnchorHiddenBatchTransform:
    """Action-free RLDS transform for offline anchor cache building."""

    def __init__(
        self,
        base_tokenizer: Any,
        image_transform: Any,
        prompt_builder_fn: type[PromptBuilder],
        prompts: Optional[Sequence[str]] = None,
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.prompts = list(prompts) if prompts else None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        instruction = rlds_batch["task"]["language_instruction"].decode().lower()

        if self.prompts:
            template = self.prompts[0]
            prompt_text = template.format(instruction=instruction)
        else:
            prompt_text = _make_prompt(self.prompt_builder_fn, instruction)

        input_ids = self.base_tokenizer(prompt_text, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        pixel_values = self.image_transform(img)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "instruction": instruction,
        }


def _anchor_collate(instances: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    input_ids = [instance["input_ids"] for instance in instances]
    attention_mask = [instance["attention_mask"] for instance in instances]
    pixel_values = [instance["pixel_values"] for instance in instances]
    instructions = [instance["instruction"] for instance in instances]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=False)

    if isinstance(pixel_values[0], dict):
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
    else:
        pixel_values = torch.stack(pixel_values)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "instructions": instructions,
    }


def _build_image_token_mask(attention_mask: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros((batch_size, seq_len + num_image_tokens), dtype=torch.bool, device=attention_mask.device)
    mask[:, 1 : 1 + num_image_tokens] = True
    return mask


def _expand_attention_mask(attention_mask: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
    prefix = attention_mask[:, :1]
    suffix = attention_mask[:, 1:]
    mid = torch.ones(
        (attention_mask.shape[0], num_image_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([prefix, mid, suffix], dim=1)


def _pool_hidden(
    hidden_states: torch.Tensor,
    pooling: str,
    image_token_mask: torch.Tensor,
    expanded_attention_mask: torch.Tensor,
) -> torch.Tensor:
    if pooling == "mean_image_tokens":
        mask = image_token_mask
    elif pooling == "mean_masked":
        mask = expanded_attention_mask.bool()
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    masked = hidden_states * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return masked.sum(dim=1) / denom


@dataclass
class BuildAnchorHiddenCacheConfig:
    # Teacher path aliases
    teacher_checkpoint: Optional[Path] = None
    teacher_model_path: Optional[str] = None
    teacher_policy_path: Optional[str] = None

    # Dataset config (OpenVLA RLDS style)
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset: str = "libero_90"
    dataset_repo_id: Optional[str] = None
    data_config: Optional[str] = None
    split: str = "train"
    image_aug: bool = False
    shuffle_buffer_size: int = 100_000

    # Output and sampling
    output_dir: Path = Path("cache/craft/hidden")
    num_samples: int = 10000
    shard_size: int = 2048
    seed: int = 7

    prompts_file: Optional[Path] = None

    hidden_layer: int = -2
    pooling: str = "mean_image_tokens"
    dtype: str = "float16"

    batch_size: int = 8

    hf_token: Union[str, Path] = Path(".hf_token")


def _resolve_teacher_path(cfg: BuildAnchorHiddenCacheConfig) -> str:
    if cfg.teacher_checkpoint is not None:
        return str(cfg.teacher_checkpoint)
    if cfg.teacher_model_path is not None:
        return cfg.teacher_model_path
    if cfg.teacher_policy_path is not None:
        return cfg.teacher_policy_path
    raise ValueError("Please provide one of teacher_checkpoint / teacher_model_path / teacher_policy_path")


def _load_prompts(prompts_file: Optional[Path]) -> Optional[List[str]]:
    if prompts_file is None:
        return None
    lines = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
    return lines if lines else None


@draccus.wrap()
def build_anchor_hidden_cache(cfg: BuildAnchorHiddenCacheConfig) -> None:
    torch.manual_seed(cfg.seed)
    cache_dtype = _parse_dtype(cfg.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("Offline hidden cache builder expects CUDA for teacher forward")

    device = torch.device("cuda", torch.cuda.current_device())
    teacher_path = _resolve_teacher_path(cfg)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # Load frozen teacher theta_0
    if cfg.teacher_checkpoint is not None:
        teacher = load_vla(teacher_path, hf_token=hf_token, load_for_training=False)
    else:
        teacher = load(teacher_path, hf_token=hf_token, load_for_training=False)
    teacher.eval()
    teacher = teacher.to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    tokenizer = teacher.llm_backbone.get_tokenizer()
    prompts = _load_prompts(cfg.prompts_file)

    # Build RLDS dataset via baseline path (same core dataset class/transform stack)
    anchor_transform = AnchorHiddenBatchTransform(
        base_tokenizer=tokenizer,
        image_transform=teacher.vision_backbone.get_image_transform(),
        prompt_builder_fn=teacher.llm_backbone.prompt_builder_fn,
        prompts=prompts,
    )
    anchor_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset,
        anchor_transform,
        resize_resolution=teacher.vision_backbone.default_image_resolution[1:],
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=(cfg.split == "train"),
        image_aug=cfg.image_aug,
    )

    dataloader = DataLoader(
        anchor_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=lambda instances: _anchor_collate(instances, tokenizer.pad_token_id),
        num_workers=0,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: List[Dict[str, Any]] = []
    shard_index = 0
    saved = 0
    num_image_tokens = teacher.vision_backbone.num_patches

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            input_ids = _maybe_append_empty_llama_token(input_ids, tokenizer)

            attention_mask = batch["attention_mask"].to(device)
            if attention_mask.shape[1] != input_ids.shape[1]:
                pad_len = input_ids.shape[1] - attention_mask.shape[1]
                if pad_len > 0:
                    pad = torch.ones((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype, device=device)
                    attention_mask = torch.cat([attention_mask, pad], dim=1)

            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(device)

            output: CausalLMOutputWithPast = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None:
                raise RuntimeError("Teacher forward did not return hidden_states; cannot build hidden cache")

            hidden = output.hidden_states[cfg.hidden_layer]
            image_token_mask = _build_image_token_mask(attention_mask, num_image_tokens)
            expanded_attention_mask = _expand_attention_mask(attention_mask, num_image_tokens)
            pooled = _pool_hidden(hidden, cfg.pooling, image_token_mask, expanded_attention_mask)

            for row in range(input_ids.shape[0]):
                model_inputs = {
                    "input_ids": input_ids[row].detach(),
                    "attention_mask": attention_mask[row].detach(),
                    "multimodal_indices": torch.tensor([0], dtype=torch.long, device=device),
                    "image_token_mask": image_token_mask[row].detach(),
                }

                if isinstance(pixel_values, dict):
                    model_inputs["pixel_values"] = {k: v[row].detach() for k, v in pixel_values.items()}
                else:
                    model_inputs["pixel_values"] = pixel_values[row].detach()

                model_inputs = _cast_floats(_move_to_cpu(model_inputs), cache_dtype)
                target_features = _cast_floats(_move_to_cpu(pooled[row]), cache_dtype)

                shard_records.append(
                    {
                        "model_inputs": model_inputs,
                        "target_features": target_features,
                        "meta": {
                            "layer": cfg.hidden_layer,
                            "pooling": cfg.pooling,
                            "dtype": cfg.dtype,
                            "teacher": teacher_path,
                            "split": cfg.split,
                        },
                    }
                )
                saved += 1

                if len(shard_records) >= cfg.shard_size:
                    shard_path = cfg.output_dir / f"hidden_anchor_shard_{shard_index:06d}.pt"
                    torch.save(
                        {
                            "records": shard_records,
                            "meta": {
                                "schema_version": 1,
                                "retention_mode": "hidden",
                                "pooling": cfg.pooling,
                                "hidden_layer": cfg.hidden_layer,
                                "dtype": cfg.dtype,
                            },
                        },
                        shard_path,
                    )
                    shard_records = []
                    shard_index += 1

                if saved >= cfg.num_samples:
                    break

            if saved >= cfg.num_samples:
                break

    if shard_records:
        shard_path = cfg.output_dir / f"hidden_anchor_shard_{shard_index:06d}.pt"
        torch.save(
            {
                "records": shard_records,
                "meta": {
                    "schema_version": 1,
                    "retention_mode": "hidden",
                    "pooling": cfg.pooling,
                    "hidden_layer": cfg.hidden_layer,
                    "dtype": cfg.dtype,
                },
            },
            shard_path,
        )

    overwatch.info(f"Saved {saved} anchor samples to {cfg.output_dir}")


if __name__ == "__main__":
    build_anchor_hidden_cache()
