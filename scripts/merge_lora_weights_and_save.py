
"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base VLA-Adapter model. Saves the final checkpoint in the same directory.

Usage:
    python scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla

# Check if NPU is available
try:
    import torch_npu
    USE_NPU = True
except ImportError:
    USE_NPU = False



@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter
    vlm_path: Union[str, Path] = "" 
    use_minivla: bool = False                        # 


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if cfg.use_minivla:
        hf_token = ''
        vlm = load_vla(
            cfg.vlm_path,
            hf_token=hf_token,
            load_for_training=True,
            )
        hf_assets_dir = Path(__file__).resolve().parents[1] / "prismatic" / "extern" / "hf"
        config = AutoConfig.from_pretrained(str(hf_assets_dir), trust_remote_code=False)
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        
        # For NPU: load without device_map and low_cpu_mem_usage to avoid meta tensor issues
        if USE_NPU:
            print("Loading to CPU (NPU compatibility mode - no device_map)")
            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.base_checkpoint,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
            )
        else:
            # For CUDA: use low_cpu_mem_usage for efficiency
            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.base_checkpoint,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )

    # Load LoRA weights and merge into base model
    print("Loading LoRA adapter...")
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter"))
    
    # Move to NPU for faster merging (if using NPU)
    if USE_NPU:
        # Support ASCEND_VISIBLE_DEVICES or ASCEND_RT_VISIBLE_DEVICES environment variable
        npu_id = int(os.environ.get("ASCEND_VISIBLE_DEVICES", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")))
        device = torch.device(f"npu:{npu_id}" if torch.npu.is_available() else "cpu")
        print(f"Moving model to {device} for faster merging...")
        merged_vla = merged_vla.to(device)
    
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = merged_vla.merge_and_unload()
    print(f"Merging complete! Time elapsed (sec): {time.time() - start_time}")
    
    # Move back to CPU for saving (for compatibility)
    if USE_NPU:
        print("Moving merged model back to CPU for saving...")
        merged_vla = merged_vla.cpu()
    
    # Save merged model
    print("Saving merged model...")
    save_start = time.time()
    merged_vla.save_pretrained(cfg.lora_finetuned_checkpoint_dir)
    print(f"Saving complete! Time elapsed (sec): {time.time() - save_start}")
    
    print(f"\nTotal time elapsed (sec): {time.time() - start_time}")
    print(f"Saved merged model checkpoint at:\n{cfg.lora_finetuned_checkpoint_dir}")


if __name__ == "__main__":
    main()
