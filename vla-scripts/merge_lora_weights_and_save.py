
"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base VLA-Adapter model. Saves the final checkpoint in the same directory.

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/

For the "MiniVLM" (Prismatic-VLM -> OpenVLA HF wrapper) workflow used by this repo's local pretrained models, run:
    python vla-scripts/merge_lora_weights_and_save.py \
        --use_minivlm True \
        --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
        --config_file_path pretrained_models/configs \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla



@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (HF repo id or local dir)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter (`lora_adapter/`)

    # MiniVLM (Prismatic-VLM) -> OpenVLA HF wrapper reconstruction (local, no download if cached/offline)
    config_file_path: Union[str, Path] = "pretrained_models/configs"
    vlm_path: Union[str, Path] = ""
    use_minivlm: bool = False
    # Backward-compat typo (older scripts used `use_minivla`)
    use_minivla: bool = False

    # Output
    output_dir: Optional[Union[str, Path]] = None            # Where to save merged HF checkpoint (default: lora_finetuned_checkpoint_dir)
    device: str = "cpu"                                      # 'cpu' or 'cuda'


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    use_minivlm = bool(getattr(cfg, "use_minivlm", False) or getattr(cfg, "use_minivla", False))

    def _rename_state_dict_keys(state_dict, replace_map):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old, new in replace_map:
                if old in new_key:
                    new_key = new_key.replace(old, new)
            new_state_dict[new_key] = value
        return new_state_dict

    if use_minivlm:
        vlm_path = str(cfg.vlm_path)
        if not vlm_path:
            raise ValueError("`--vlm_path` is required when `--use_minivlm True`.")

        hf_token = ""
        # Mirror `vla-scripts/finetune.py` logic: Prismatic-VLM uses `load`, OpenVLA uses `load_vla`.
        if "prism-qwen25-extra-dinosiglip-224px-0_5b" in vlm_path:
            vlm = load(vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(vlm_path, hf_token=hf_token, load_for_training=True)

        config = AutoConfig.from_pretrained(str(cfg.config_file_path), trust_remote_code=False)
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)

        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        raw_state_dict = _rename_state_dict_keys(vlm.state_dict(), replace_map)
        missing_keys, unexpected_keys = vla.load_state_dict(raw_state_dict, strict=False)
        if missing_keys:
            print(f"[merge] Missing keys while loading base VLM weights: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[merge] Unexpected keys while loading base VLM weights: {len(unexpected_keys)}")

    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    device = torch.device(cfg.device)
    vla = vla.to(device)

    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    adapter_dir = os.path.join(str(cfg.lora_finetuned_checkpoint_dir), "lora_adapter")
    merged_vla = PeftModel.from_pretrained(vla, adapter_dir).to(device)
    merged_vla = merged_vla.merge_and_unload()
    out_dir = str(cfg.output_dir) if cfg.output_dir is not None else str(cfg.lora_finetuned_checkpoint_dir)
    merged_vla.save_pretrained(out_dir, safe_serialization=True)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{out_dir}")


if __name__ == "__main__":
    main()
