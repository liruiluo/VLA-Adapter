"""
openvla.py

Register OpenVLA and VLAAdapter models to Hugging Face AutoClasses. [Add][fancy_vla]
"""
from typing import Tuple

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from ..extern.hf.configuration_prismatic import OpenVLAConfig
from ..extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from ..extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla():
    """Register original OpenVLA model (token-based prediction only)."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_openvla(
    vlm_path: str, 
    config_file_path: str, 
    device: torch.device, 
    use_lora: bool,
    use_quantization: bool,
) -> Tuple[PrismaticProcessor, OpenVLAForActionPrediction]:
    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if use_quantization:
        assert use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # config = OpenVLAConfig.from_pretrained(config_file_path)
    # processor = PrismaticProcessor.from_pretrained(vlm_path, trust_remote_code=False)
    # vla = OpenVLAForActionPrediction.from_pretrained(
    #     vlm_path, 
    #     config=config,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=quantization_config if use_lora else None,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    register_openvla()
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        vlm_path,        
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if use_lora else None,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )

    if use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device)
    return processor, vla