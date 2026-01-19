"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.

=== ACTION_TOKEN_BEGIN_IDX Auto-Detection ===

This module automatically detects whether you're using:
- OpenVLA (7B, Llama2 tokenizer): ACTION_TOKEN_BEGIN_IDX = 31743
- VLA-Adapter (0.5B, Qwen2.5 tokenizer): ACTION_TOKEN_BEGIN_IDX = 151386

Detection priority:
1. Environment variable: export VLA_MODEL_TYPE=OPENVLA (or VLA_ADAPTER)
2. Command-line arguments: Detects script name (finetune_openvla.py vs finetune.py)
3. Default: VLA-Adapter (for backward compatibility)

Manual override examples:
    # For OpenVLA
    export VLA_MODEL_TYPE=OPENVLA
    python scripts/finetune_openvla.py ...
    
    # For VLA-Adapter
    export VLA_MODEL_TYPE=VLA_ADAPTER
    python scripts/finetune.py ...
    
    # Auto-detection (recommended)
    python scripts/finetune_openvla.py ...  # Automatically uses OpenVLA
    python scripts/finetune.py ...          # Automatically uses VLA-Adapter
"""
import os
import sys
from enum import Enum

# Token constants
IGNORE_INDEX = -100
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64

# Model-specific action token begin index
VLA_ADAPTER_ACTION_TOKEN_BEGIN_IDX = 151386  # VLA-Adapter (Qwen2.5-0.5B tokenizer)
OPENVLA_ACTION_TOKEN_BEGIN_IDX = 31743       # OpenVLA (Llama2-7B tokenizer)


def detect_model_type():
    """
    Automatically detect which model is being used based on:
    1. Environment variable VLA_MODEL_TYPE (highest priority)
    2. Command-line arguments (script name or explicit model path)
    3. Default to VLA-Adapter for backward compatibility
    
    Returns:
        str: "OPENVLA" or "VLA_ADAPTER"
    """
    # Priority 1: Check environment variable
    env_model = os.environ.get("VLA_MODEL_TYPE", "").upper()
    if env_model in ["OPENVLA", "VLA_ADAPTER", "VLA-ADAPTER"]:
        return "OPENVLA" if env_model == "OPENVLA" else "VLA_ADAPTER"
    
    # Priority 2: Check command-line arguments
    cmd_args = " ".join(sys.argv).lower()
    
    # Check for OpenVLA indicators (more specific patterns first)
    openvla_indicators = [
        "finetune_openvla",      # Script name
        "openvla-7b",            # Model name
        "openvla/openvla",       # HF model path
        "--use_hf_model true",   # Explicit HF model flag
    ]
    
    vla_adapter_indicators = [
        "finetune.py",           # Original VLA-Adapter script
        "vla-adapter",           # Model name
        "vla_adapter",           # Model name (underscore)
        "qwen",                  # Qwen model
        "0.5b",                  # 0.5B model size
        "0_5b",                  # Alternative format
    ]
    
    # Check OpenVLA indicators
    for indicator in openvla_indicators:
        if indicator in cmd_args:
            return "OPENVLA"
    
    # Check VLA-Adapter indicators
    for indicator in vla_adapter_indicators:
        if indicator in cmd_args:
            return "VLA_ADAPTER"
    
    # Priority 3: Default to VLA-Adapter for backward compatibility
    return "VLA_ADAPTER"


# Detect model type and set ACTION_TOKEN_BEGIN_IDX accordingly
MODEL_TYPE = detect_model_type()
ACTION_TOKEN_BEGIN_IDX = (
    OPENVLA_ACTION_TOKEN_BEGIN_IDX if MODEL_TYPE == "OPENVLA" 
    else VLA_ADAPTER_ACTION_TOKEN_BEGIN_IDX
)

# Print detection result (only once, for rank 0 in distributed training)
_printed_detection = False
if not _printed_detection:
    import torch.distributed as dist
    try:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Constants] Detected model type: {MODEL_TYPE}")
            print(f"[Constants] ACTION_TOKEN_BEGIN_IDX set to: {ACTION_TOKEN_BEGIN_IDX}")
            _printed_detection = True
    except:
        # If distributed training is not initialized, just print
        print(f"[Constants] Detected model type: {MODEL_TYPE}")
        print(f"[Constants] ACTION_TOKEN_BEGIN_IDX set to: {ACTION_TOKEN_BEGIN_IDX}")
        _printed_detection = True


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "calvin" in cmd_args:
        return "CALVIN"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]
# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")

