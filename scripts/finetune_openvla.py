"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
try:
    import torch_npu
    USE_NPU = True
except ImportError:
    USE_NPU = False
import torch.distributed as dist
from tqdm import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import VLA-Adapter loader if needed
try:
    from prismatic.vla import load_vla_adapter
except ImportError:
    load_vla_adapter = None

# Import FSDP training utilities
try:
    from prismatic.training import get_train_strategy, VLAMetrics
    from prismatic.util import set_global_seed
    from prismatic.overwatch import initialize_overwatch
    from prismatic.vla import get_vla_dataset_and_collator
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    get_train_strategy = None
    VLAMetrics = None
    set_global_seed = None
    initialize_overwatch = None
    get_vla_dataset_and_collator = None


# === Configuration ===

@dataclass
class FinetuneConfig:
    # fmt: off
    # Model Paths and Loading Mode
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub or local)
    use_hf_model: bool = True                                       # True=HF AutoClasses, False=VLA-Adapter loading
    config_file_path: str = "prismatic/extern/hf"                   # Only used when use_hf_model=False
    num_images_in_input: int = 1                                    # Number of images in input (for VLA-Adapter mode)
    use_flash_attention_2: Optional[bool] = None                     # Whether to use flash attention 2

    # Training Mode Selection
    training_mode: str = "lora"                                     # "lora" or "full"
    use_fsdp: bool = False                                          # Whether to use FSDP (only for full training mode)
    train_strategy: str = "fsdp-shard-grad-op"                      # FSDP strategy: "fsdp-shard-grad-op" or "fsdp-full-shard"

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size (per device)
    global_batch_size: Optional[int] = None                         # Global batch size (for FSDP, computed if None)
    per_device_batch_size: Optional[int] = None                     # Per device batch size (for FSDP, uses batch_size if None)
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    epochs: Optional[int] = None                                    # Number of epochs (for FSDP, uses max_steps if None)
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    weight_decay: float = 0.01                                      # Weight decay (for FSDP)
    max_grad_norm: float = 1.0                                     # Max gradient norm (for FSDP)
    lr_scheduler_type: str = "cosine"                               # LR scheduler type: "cosine" or "constant" (for FSDP)
    warmup_ratio: float = 0.01                                     # Warmup ratio (for FSDP)
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    
    # Data Processing Parameters
    use_minivlm: bool = True                                        # Whether to use minivlm data processing
                                                                    #   (True for OpenVLA 7B, False for VLA-Adapter 0.5B)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Full Training Parameters (for FSDP)
    freeze_vision_backbone: bool = False                            # Whether to freeze vision backbone
    freeze_llm_backbone: bool = False                               # Whether to freeze LLM backbone
    unfreeze_last_llm_layer: bool = True                            # Whether to unfreeze last LLM layer
    enable_gradient_checkpointing: bool = True                      # Whether to enable gradient checkpointing
    enable_mixed_precision_training: bool = True                    # Whether to enable mixed precision training
    reduce_in_full_precision: bool = False                          # Whether to reduce in full precision (for FSDP)

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    trackers: tuple = ("jsonl", "wandb")                            # Trackers to use (for FSDP)

    # Random seed
    seed: int = 7                                                   # Random seed for reproducibility

    # fmt: on


# === Utilities ===

def add_fsdp_support_to_hf_model(vla, cfg: FinetuneConfig):
    """
    Add FSDP-required attributes and methods to HF-loaded OpenVLA model.
    This makes the HF model compatible with VLA-Adapter's FSDP training code.
    
    Args:
        vla: HuggingFace loaded OpenVLA model
        cfg: Configuration object
        
    Returns:
        vla: Model with FSDP support added
    """
    from functools import partial
    from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    
    # Add module keys attributes (required by FSDP training strategy)
    vla.all_module_keys = ["vision_backbone", "language_model", "projector"]
    vla.trainable_module_keys = []
    
    # Determine trainable modules based on configuration
    if cfg.training_mode == "full":
        if not cfg.freeze_vision_backbone:
            vla.trainable_module_keys.append("vision_backbone")
        if not cfg.freeze_llm_backbone:
            vla.trainable_module_keys.append("language_model")
        vla.trainable_module_keys.append("projector")
    else:  # LoRA or frozen
        # For LoRA, trainable keys will be the LoRA adapters
        vla.trainable_module_keys = []
    
    # Add FSDP wrapping policy method
    def get_fsdp_wrapping_policy(self):
        """Return FSDP wrapping policy for OpenVLA model."""
        from prismatic.extern.hf.modeling_prismatic import (
            PrismaticProjector,
            PrismaticVisionBackbone
        )
        
        # Determine LLM decoder layer class based on the language model
        llm_decoder_layer_cls = None
        if hasattr(self.language_model, "model") and hasattr(self.language_model.model, "layers"):
            if len(self.language_model.model.layers) > 0:
                first_layer = self.language_model.model.layers[0]
                llm_decoder_layer_cls = type(first_layer)
        
        # Fallback to common types if detection fails
        if llm_decoder_layer_cls is None:
            llm_decoder_layer_cls = (LlamaDecoderLayer, MistralDecoderLayer)
        
        # Vision backbone policy (wrap the vision backbone module)
        vision_policy = partial(
            _module_wrap_policy,
            module_classes={PrismaticVisionBackbone},
        )
        
        # LLM policy (wrap each decoder layer)
        llm_policy = partial(
            _module_wrap_policy,
            module_classes={llm_decoder_layer_cls} if not isinstance(llm_decoder_layer_cls, tuple) else set(llm_decoder_layer_cls),
        )
        
        # Projector policy (wrap the projector module)
        projector_policy = partial(
            _module_wrap_policy,
            module_classes={PrismaticProjector},
        )
        
        # Combine all policies
        return partial(
            _or_policy,
            policies=[vision_policy, llm_policy, projector_policy],
        )
    
    # Bind the method to the model instance
    import types
    vla.get_fsdp_wrapping_policy = types.MethodType(get_fsdp_wrapping_policy, vla)
    
    # Add llm_transformer_layer_cls attribute for gradient checkpointing
    if hasattr(vla.language_model, "model") and hasattr(vla.language_model.model, "layers"):
        if len(vla.language_model.model.layers) > 0:
            vla.llm_transformer_layer_cls = type(vla.language_model.model.layers[0])
        else:
            # Fallback
            vla.llm_transformer_layer_cls = (LlamaDecoderLayer, MistralDecoderLayer)
    else:
        vla.llm_transformer_layer_cls = (LlamaDecoderLayer, MistralDecoderLayer)
    
    # Add attribute aliases to make HF model compatible with VLA-Adapter's FSDP strategy
    # VLA-Adapter expects: llm_backbone, but HF uses: language_model
    
    # Create a wrapper class that acts like llm_backbone but wraps language_model
    class LLMBackboneWrapper:
        """Wrapper to make HF's language_model compatible with VLA-Adapter's llm_backbone interface."""
        def __init__(self, language_model, transformer_layer_cls):
            # Store as _language_model to avoid infinite recursion in __getattr__
            object.__setattr__(self, '_language_model', language_model)
            object.__setattr__(self, 'transformer_layer_cls', transformer_layer_cls)
        
        def __getattr__(self, name):
            # Forward all other attributes to the wrapped language_model
            return getattr(self._language_model, name)
        
        def __setattr__(self, name, value):
            # Forward attribute setting to the wrapped language_model
            if name in ('_language_model', 'transformer_layer_cls'):
                object.__setattr__(self, name, value)
            else:
                setattr(self._language_model, name, value)
        
        def __call__(self, *args, **kwargs):
            # Forward calls to the wrapped language_model
            return self._language_model(*args, **kwargs)
    
    # Wrap the language_model with transformer_layer_cls
    vla.llm_backbone = LLMBackboneWrapper(vla.language_model, vla.llm_transformer_layer_cls)
    
    # Add half_precision_dtype attribute to vision_backbone (required by FSDP)
    if not hasattr(vla.vision_backbone, 'half_precision_dtype'):
        vla.vision_backbone.half_precision_dtype = torch.bfloat16
    
    print("[INFO] Added FSDP support to HF-loaded OpenVLA model")
    print(f"[INFO] All module keys: {vla.all_module_keys}")
    print(f"[INFO] Trainable module keys: {vla.trainable_module_keys}")
    print(f"[INFO] LLM transformer layer class: {vla.llm_transformer_layer_cls}")
    
    return vla


def load_model_for_finetuning(cfg: FinetuneConfig, device_id: int, use_fsdp: bool = False):
    """
    Load model for fine-tuning using either HF AutoClasses or VLA-Adapter method.
    
    Args:
        cfg: Configuration object
        device_id: Device ID for model placement
        use_fsdp: Whether using FSDP (if True, load to CPU first)
        
    Returns:
        processor: Processor for tokenization and image processing
        vla: VLA model
    """
    if cfg.use_hf_model:
        # OpenVLA way: Use HF AutoClasses
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)
        
        # Quantization config for LoRA
        quantization_config = None
        if cfg.use_quantization and cfg.use_lora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        
        # Device map configuration
        # - FSDP: Load to CPU, let FSDP handle device placement
        # - NPU without FSDP: Use device_map to avoid CUDA-related issues
        # - CUDA without FSDP: Load to CPU first, then move to device
        device_map = None
        if use_fsdp:
            # FSDP will handle device placement, load to CPU
            device_map = None
        elif USE_NPU and not cfg.use_quantization:
            # For NPU, use device_map to place model directly
            device_map = {"": device_id}
        
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        
        # Device placement (only for non-FSDP, non-NPU cases)
        if cfg.use_quantization:
            vla = prepare_model_for_kbit_training(vla)
        elif not use_fsdp and not USE_NPU:
            # Only call .to() for CUDA devices in non-FSDP mode
            vla = vla.to(device_id)
        
        # Add FSDP support for HF models if using FSDP
        if use_fsdp:
            vla = add_fsdp_support_to_hf_model(vla, cfg)
            
    else:
        # VLA-Adapter way
        if load_vla_adapter is None:
            raise ImportError("VLA-Adapter loader not available. Install VLA-Adapter dependencies.")
        
        # For FSDP, load to CPU; otherwise load to specific device
        if use_fsdp:
            device = torch.device("cpu")
            print("[INFO] Loading model to CPU for FSDP initialization...")
        else:
            # device_id might be int or torch.device, normalize it
            if isinstance(device_id, torch.device):
                device = device_id
            else:
                if USE_NPU:
                    device = torch.device(f"npu:{device_id}")
                else:
                    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        processor, vla, _ = load_vla_adapter(
            vlm_path=cfg.vla_path,
            config_file_path=cfg.config_file_path,
            device=device,
            num_images_in_input=cfg.num_images_in_input,
            use_flash_attention_2=False if USE_NPU else cfg.use_flash_attention_2,
        )
    
    return processor, vla


def determine_training_stage(cfg: FinetuneConfig) -> str:
    """
    Determine training stage based on freezing configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        stage: Training stage string
    """
    if not cfg.freeze_vision_backbone and not cfg.freeze_llm_backbone:
        return "vla-full-train"  # Full fine-tuning
    elif cfg.freeze_vision_backbone and not cfg.freeze_llm_backbone:
        return "vla-train"  # Frozen vision encoder
    elif not cfg.freeze_vision_backbone and cfg.freeze_llm_backbone:
        assert cfg.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        return "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.freeze_vision_backbone and cfg.freeze_llm_backbone:
        assert cfg.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        return "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            f"Weight freezing configuration not supported. "
            f"freeze_vision_backbone: {cfg.freeze_vision_backbone}, "
            f"freeze_llm_backbone: {cfg.freeze_llm_backbone}, "
            f"unfreeze_last_llm_layer: {cfg.unfreeze_last_llm_layer}"
        )


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    print(f"Training mode: {cfg.training_mode}, Use FSDP: {cfg.use_fsdp}, Use HF Model: {cfg.use_hf_model}")

    # [Validate] Ensure GPU/NPU Available & Set Device / Distributed Context
    if USE_NPU:
        assert torch.npu.is_available(), "Fine-tuning assumes at least one NPU is available!"
    else:
        assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    
    # Initialize distributed state
    overwatch = None
    if cfg.training_mode == "full" and cfg.use_fsdp and FSDP_AVAILABLE and initialize_overwatch:
        # Use overwatch for FSDP (initializes torch.distributed automatically)
        overwatch = initialize_overwatch(__name__)
        device_index = overwatch.local_rank()
        distributed_state = None  # overwatch handles this
        is_main_process = overwatch.is_rank_zero()
    else:
        # Use PartialState for DDP/LoRA training
        distributed_state = PartialState()
        device_index = distributed_state.local_process_index
        is_main_process = distributed_state.is_main_process
    
    # Set device and create torch.device object
    if USE_NPU:
        torch.npu.set_device(device_index)
        torch.npu.empty_cache()
        device_id = torch.device(f"npu:{device_index}")
    else:
        torch.cuda.set_device(device_index)
        torch.cuda.empty_cache()
        device_id = torch.device(f"cuda:{device_index}")

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.training_mode == "lora" and cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    elif cfg.training_mode == "full":
        exp_id += "+full"
        if cfg.use_fsdp:
            exp_id += f"+fsdp-{cfg.train_strategy}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    if not cfg.use_hf_model:
        exp_id += "+adapter"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    if cfg.training_mode == "full" and cfg.use_fsdp:
        os.makedirs(run_dir / "checkpoints", exist_ok=True)

    # Load model using appropriate method
    # For FSDP, load to CPU first; FSDP will handle device placement
    use_fsdp_for_loading = cfg.training_mode == "full" and cfg.use_fsdp and FSDP_AVAILABLE
    processor, vla = load_model_for_finetuning(cfg, device_index, use_fsdp=use_fsdp_for_loading)

    # Apply LoRA if needed (only for LoRA training mode)
    if cfg.training_mode == "lora" and cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Handle FSDP vs DDP wrapping
    train_strategy = None
    if cfg.training_mode == "full" and cfg.use_fsdp and FSDP_AVAILABLE:
        # FSDP training path
        # Validate model is in full precision for FSDP
        for param in vla.parameters():
            if param.dtype != torch.float32:
                # Convert to float32 for FSDP (will be converted to bfloat16 during training)
                param.data = param.data.float()
        
        # Determine training stage
        stage = determine_training_stage(cfg)
        
        # Freeze backbones if needed (for OpenVLA models that support it)
        if hasattr(vla, 'freeze_backbones'):
            vla.freeze_backbones(stage)
        
        # Compute batch sizes
        if cfg.global_batch_size is None:
            if overwatch and hasattr(overwatch, 'world_size'):
                world_size = overwatch.world_size()
            else:
                # Fallback: try to get from torch.distributed
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                else:
                    world_size = 1
            cfg.global_batch_size = cfg.batch_size * world_size
        if cfg.per_device_batch_size is None:
            cfg.per_device_batch_size = cfg.batch_size
        
        # Create train strategy
        worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True) if set_global_seed else None
        # FSDP needs integer device index, not torch.device object
        train_strategy = get_train_strategy(
            train_strategy=cfg.train_strategy,
            vlm=vla,
            device_id=device_index,
            stage=stage,
            epochs=cfg.epochs if cfg.epochs is not None else 1,
            max_steps=cfg.max_steps,
            global_batch_size=cfg.global_batch_size,
            per_device_batch_size=cfg.per_device_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            enable_gradient_checkpointing=cfg.enable_gradient_checkpointing,
            enable_mixed_precision_training=cfg.enable_mixed_precision_training,
            reduce_in_full_precision=cfg.reduce_in_full_precision,
            worker_init_fn=worker_init_fn,
        )
    else:
        # DDP training path (for LoRA or non-FSDP full training)
        vla = DDP(vla, device_ids=[device_index], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset
    # Get image sizes from model config
    if hasattr(vla, 'module'):
        # DDP wrapped model
        image_sizes = tuple(vla.module.config.image_sizes)
        prompt_builder_fn = PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder
    elif hasattr(vla, 'config'):
        # Direct model
        image_sizes = tuple(vla.config.image_sizes)
        prompt_builder_fn = PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder
    else:
        # Fallback
        image_sizes = (224, 224)
        prompt_builder_fn = PurePromptBuilder

    # For FSDP with HF models, we need to use RLDSDataset for compatibility
    # Only use get_vla_dataset_and_collator for FSDP with VLA-Adapter models
    use_rlds_dataset = True
    if cfg.training_mode == "full" and cfg.use_fsdp and FSDP_AVAILABLE and not cfg.use_hf_model:
        # FSDP + VLA-Adapter model: use OpenVLA's dataset loader
        if get_vla_dataset_and_collator:
            vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
                cfg.data_root_dir,
                cfg.dataset_name,
                image_transform=processor.image_processor.apply_transform,
                tokenizer=processor.tokenizer,
                prompt_builder_fn=prompt_builder_fn,
                default_image_resolution=image_sizes,
                shuffle_buffer_size=cfg.shuffle_buffer_size,
                image_aug=cfg.image_aug,
            )
            dataloader = None  # FSDP strategy handles dataloader creation
            use_rlds_dataset = False
    
    if use_rlds_dataset:
        # FSDP + HF model OR DDP/LoRA: use standard RLDS dataset for compatibility
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=prompt_builder_fn,
            use_minivlm=cfg.use_minivlm,
        )
        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=image_sizes,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        
        # Create Collator
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        
        # Create DataLoader only for non-FSDP modes (FSDP strategy creates its own)
        if cfg.training_mode == "full" and cfg.use_fsdp and FSDP_AVAILABLE:
            dataloader = None  # FSDP strategy handles dataloader creation
        else:
            # DDP or LoRA mode: create dataloader
            dataloader = DataLoader(
                vla_dataset,
                batch_size=cfg.batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
            )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Initialize Logging =>> W&B
    if is_main_process:
        try:
            wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")
            print("[WARNING] Continuing training in offline mode...")

    # Training loop: FSDP vs DDP
    # Validate FSDP availability if requested
    if cfg.training_mode == "full" and cfg.use_fsdp:
        if not FSDP_AVAILABLE or train_strategy is None:
            print("WARNING: FSDP requested but not available. Falling back to DDP training.")
            cfg.use_fsdp = False
            cfg.training_mode = "lora"  # Use DDP path
            # Re-wrap with DDP if not already wrapped
            if not isinstance(vla, DDP):
                vla = DDP(vla, device_ids=[device_index], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    if cfg.training_mode == "full" and cfg.use_fsdp and train_strategy is not None:
        # FSDP Training Path with custom training loop
        # Run setup (wraps model with FSDP, sets up optimizer, etc.)
        train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
        
        # Custom training loop (instead of run_vla_training)
        import time
        
        # Create DataLoader (already imported at top of file)
        dataloader = DataLoader(
            vla_dataset,
            batch_size=train_strategy.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=train_strategy.worker_init_fn,
        )
        
        # Training metrics
        global_step = 0
        start_time = time.time()
        
        # Use the already initialized overwatch
        overwatch_train = overwatch
        
        # Training loop
        print(f"Starting FSDP training for {cfg.max_steps} steps...")
        with tqdm(
            total=cfg.max_steps,
            desc=f"=>> [Epoch 000] Global Step {global_step:06d}",
            leave=False,
            disable=not overwatch_train.is_rank_zero(),
        ) as progress:
            train_strategy.vlm.train()
            train_strategy.optimizer.zero_grad()
            
            for batch in dataloader:
                # Forward pass with automatic mixed precision
                device_type = "npu" if USE_NPU else "cuda"
                with torch.autocast(
                    device_type, dtype=train_strategy.mixed_precision_dtype, 
                    enabled=train_strategy.enable_mixed_precision_training
                ):
                    output = train_strategy.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                
                # Backward pass
                loss.backward()
                
                # === Compute Metrics (including next_actions) ===
                # Get num_patches
                try:
                    num_patches = train_strategy.vlm.vision_backbone.num_patches
                except AttributeError:
                    num_patches = 257  # Default for OpenVLA
                
                # Get predictions - ensure alignment between predicted and ground truth
                # output.logits: [batch, vision_tokens + text_tokens, vocab]
                # batch["labels"]: [batch, text_tokens] (no vision tokens)
                predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
                # To match predicted length: if predicted is (batch, N), labels should also be (batch, N)
                # predicted: logits[num_patches:-1] means we skip first num_patches and last 1
                # labels: we need to skip first token for autoregressive, giving us labels[1:]
                # But we need to ensure they have same length
                ground_truth_token_ids = batch["labels"][:, 1:].to(predicted_token_ids.device)
                
                # If lengths don't match, adjust ground_truth to match predicted
                if predicted_token_ids.shape[1] < ground_truth_token_ids.shape[1]:
                    ground_truth_token_ids = ground_truth_token_ids[:, :predicted_token_ids.shape[1]]
                elif predicted_token_ids.shape[1] > ground_truth_token_ids.shape[1]:
                    # This shouldn't happen, but handle it just in case
                    predicted_token_ids = predicted_token_ids[:, :ground_truth_token_ids.shape[1]]
                
                # Compute current action metrics
                # Note: constants.py now automatically sets correct ACTION_TOKEN_BEGIN_IDX based on model type
                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                curr_action_accuracy = compute_token_accuracy(
                    predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
                )
                curr_action_l1_loss = compute_actions_l1_loss(
                    action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
                )
                
                # Compute next actions metrics
                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                
                # Only compute if there are next actions in this batch
                if next_actions_mask.sum() > 0:
                    next_actions_accuracy = compute_token_accuracy(
                        predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
                    )
                    next_actions_l1_loss = compute_actions_l1_loss(
                        action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
                    )
                else:
                    next_actions_accuracy = torch.tensor(0.0, device=predicted_token_ids.device)
                    next_actions_l1_loss = torch.tensor(0.0, device=predicted_token_ids.device)
                
                # Gradient clipping and optimizer step
                train_strategy.clip_grad_norm()
                train_strategy.optimizer.step()
                train_strategy.lr_scheduler.step()
                train_strategy.optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                
                # Log metrics
                if overwatch_train.is_rank_zero() and global_step % 10 == 0:
                    step_time = (time.time() - start_time) / 10
                    start_time = time.time()
                    
                    metrics_dict = {
                        "VLA Train/Step": global_step,
                        "VLA Train/Loss": loss.item(),
                        "VLA Train/Action Token Accuracy": curr_action_accuracy.item(),
                        "VLA Train/L1 Loss": curr_action_l1_loss.item(),
                        "VLA Train/Next Actions Accuracy": next_actions_accuracy.item(),
                        "VLA Train/Next Actions L1 Loss": next_actions_l1_loss.item(),
                        "VLA Train/Learning Rate": train_strategy.lr_scheduler.get_last_lr()[0],
                        "VLA Train/Step Time": step_time,
                    }
                    
                    try:
                        wandb.log(metrics_dict, step=global_step)
                    except:
                        pass  # Ignore wandb errors
                    
                    progress.set_description(
                        f"=>> [Epoch {global_step // (len(vla_dataset) // cfg.global_batch_size):03d}] "
                        f"Global Step {global_step:06d} =>> Loss: {loss.item():.4f}, "
                        f"Action Acc: {curr_action_accuracy.item():.4f}"
                    )
                
                progress.update()
                
                # Save checkpoint
                if global_step % cfg.save_steps == 0:
                    if overwatch_train.is_rank_zero():
                        print(f"\n=>> Saving checkpoint at step {global_step}")
                    epoch = global_step // (len(vla_dataset) // cfg.global_batch_size)
                    train_strategy.save_checkpoint(
                        run_dir, global_step, epoch, loss.item(), only_trainable=False
                    )
                    if USE_NPU:
                        import torch.distributed as dist_check
                        if dist_check.is_initialized():
                            dist_check.barrier()
                
                # Check for termination
                if global_step >= cfg.max_steps:
                    break
        
        # Save final checkpoint
        if overwatch_train.is_rank_zero():
            print(f"\n=>> Training complete! Saving final checkpoint...")
        epoch = global_step // (len(vla_dataset) // cfg.global_batch_size)
        train_strategy.save_checkpoint(
            run_dir, global_step, epoch, loss.item(), only_trainable=False
        )
        
        if hasattr(overwatch, 'info'):
            overwatch.info("Done with FSDP Training")
        
    else:
        # DDP/LoRA Training Path
        # Create Optimizer =>> note that we default to a simple constant learning rate!
        trainable_params = [param for param in vla.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)


        # Train!
        with tqdm(total=cfg.max_steps, leave=False) as progress:
            vla.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("npu" if USE_NPU else "cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"].to(device_id),
                    )
                    loss = output.loss
                    
                    # Safety check: skip batch if loss is None
                    if loss is None:
                        print(f"[WARNING] Loss is None at batch {batch_idx}, skipping...")
                        continue

                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                # Get num_patches and prepare token IDs
                num_patches = vla.module.vision_backbone.featurizer.patch_embed.num_patches
                
                # CRITICAL: batch["labels"] does NOT include vision patches, only text!
                # - output.logits: [batch, num_patches + text_len, vocab_size]
                # - batch["labels"]: [batch, text_len] (no vision patches)
                # So we need to:
                # - predicted_token_ids: extract text part from logits (skip patches and last token)
                # - ground_truth_token_ids: extract from labels (skip first token for shifting)
                predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)  # Text predictions (shifted)
                ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)          # Text labels (shifted)
                
                # Debug: Print shapes for first batch
                if batch_idx == 0 and dist.get_rank() == 0:
                    print(f"\n=== DEBUG: Token Shapes ===")
                    print(f"num_patches: {num_patches}")
                    print(f"output.logits shape: {output.logits.shape}")
                    print(f"predicted_token_ids shape: {predicted_token_ids.shape}")
                    print(f"batch['labels'] shape: {batch['labels'].shape}")
                    print(f"ground_truth_token_ids shape: {ground_truth_token_ids.shape}")
                    print(f"ground_truth_token_ids sample (first 20): {ground_truth_token_ids[0][:20]}")
                    non_ignore = ground_truth_token_ids[0] != IGNORE_INDEX
                    print(f"Non-IGNORE tokens: {non_ignore.sum().item()}")
                    print(f"Unique values in ground_truth (non-IGNORE): {torch.unique(ground_truth_token_ids[0][non_ignore])[:10]}")
                
                # Get action masks
                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                
                # Debug: Print mask info for first batch
                if batch_idx == 0 and dist.get_rank() == 0:
                    print(f"current_action_mask sum: {current_action_mask.sum().item()}")
                    print(f"next_actions_mask sum: {next_actions_mask.sum().item()}")
                    print(f"ACTION_TOKEN_BEGIN_IDX: {ACTION_TOKEN_BEGIN_IDX}")
                
                # Compute current action metrics
                curr_action_accuracy = compute_token_accuracy(
                    predicted_token_ids,
                    ground_truth_token_ids,
                    mask=current_action_mask
                )
                curr_action_l1_loss = compute_actions_l1_loss(
                    action_tokenizer,
                    predicted_token_ids,
                    ground_truth_token_ids,
                    mask=current_action_mask
                )
                
                # Compute next actions metrics
                if next_actions_mask.sum() > 0:
                    next_actions_accuracy = compute_token_accuracy(
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=next_actions_mask
                    )
                    next_actions_l1_loss = compute_actions_l1_loss(
                        action_tokenizer,
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=next_actions_mask
                    )
                else:
                    # No next actions in this batch
                    next_actions_accuracy = torch.tensor(0.0, device=predicted_token_ids.device)
                    next_actions_l1_loss = torch.tensor(0.0, device=predicted_token_ids.device)

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Push Metrics to W&B (every 10 gradient steps)
                if is_main_process and gradient_step_idx % 10 == 0:
                    try:
                        wandb.log(
                            {
                                "train_loss": loss.item(),
                                "action_accuracy": curr_action_accuracy.item(),
                                "l1_loss": curr_action_l1_loss.item(),
                                "next_actions_accuracy": next_actions_accuracy.item(),
                                "next_actions_l1_loss": next_actions_l1_loss.item(),
                            },
                            step=gradient_step_idx,
                        )
                    except:
                        pass  # Ignore WandB errors
                    

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                    # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                    if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                        if is_main_process:
                            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if (cfg.use_lora and cfg.training_mode == "lora") else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.module.save_pretrained(save_dir)

                        # Wait for processor and adapter weights to be saved by main process
                        dist.barrier()

                        # Merge LoRA weights into model backbone for faster inference
                        #   =>> Note that merging is slow and can be done post-hoc to speed up training
                        if cfg.use_lora and cfg.training_mode == "lora":
                            if cfg.use_hf_model:
                                base_vla = AutoModelForVision2Seq.from_pretrained(
                                    cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
                                )
                            else:
                                # For VLA-Adapter, we need to reload the base model
                                if USE_NPU:
                                    device = torch.device(f"npu:{device_id}")
                                else:
                                    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
                                _, base_vla, _ = load_vla_adapter(
                                    vlm_path=cfg.vla_path,
                                    config_file_path=cfg.config_file_path,
                                    device=device,
                                    num_images_in_input=cfg.num_images_in_input,
                                    use_flash_attention_2=False if USE_NPU else cfg.use_flash_attention_2,
                                )
                            
                            merged_vla = PeftModel.from_pretrained(base_vla, str(adapter_dir))
                            merged_vla = merged_vla.merge_and_unload()
                            if is_main_process:
                                if cfg.save_latest_checkpoint_only:
                                    # Overwrite latest checkpoint
                                    merged_vla.save_pretrained(run_dir)

                                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                                else:
                                    # Prepare to save checkpoint in new directory
                                    checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                                    os.makedirs(checkpoint_dir, exist_ok=True)

                                    # Save dataset statistics to new directory
                                    save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                                    # Save processor and model weights to new directory
                                    processor.save_pretrained(checkpoint_dir)
                                    merged_vla.save_pretrained(checkpoint_dir)

                                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                            # Block on Main Process Checkpointing
                            dist.barrier()

                    # Stop training when max_steps is reached
                    if gradient_step_idx == cfg.max_steps:
                        print(f"Max step {cfg.max_steps} reached! Stopping training...")
                        break


if __name__ == "__main__":
    finetune()