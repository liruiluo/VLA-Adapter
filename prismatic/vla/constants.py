"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation.
"""
import os
import sys
from enum import Enum
import torch.distributed as dist

# --- 1. Define Enums & Basic Constants ---

# Token constants
IGNORE_INDEX = -100
STOP_INDEX = 0  # only for VLA-Adapter

class VLAArchitecture(str, Enum):
    OPENVLA = "openvla"          # Llama-2-7B based
    VLA_ADAPTER = "vla_adapter"  # Qwen2.5-0.5B based
    # Future extensions can be added here
    # RT2 = "rt2"

class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on

# Map Architecture to specific constants
MODEL_CONSTANTS = {
    VLAArchitecture.OPENVLA: {
        "ACTION_TOKEN_BEGIN_IDX": 31743,  # Llama2 tokenizer
    },
    VLAArchitecture.VLA_ADAPTER: {
        "ACTION_TOKEN_BEGIN_IDX": 151386, # Qwen2.5 tokenizer
    }
}

# --- 2. Model Architecture Detection Logic ---

def detect_model_type() -> VLAArchitecture:
    """
    Automatically detect which model architecture is being used.
    Priority:
    1. Environment variable VLA_MODEL_TYPE
    2. Command-line arguments inspection
    3. Default to VLA_ADAPTER
    """
    # Priority 1: Check environment variable
    env_model = os.environ.get("VLA_MODEL_TYPE", "").lower()
    if env_model == "openvla":
        return VLAArchitecture.OPENVLA
    elif env_model in ["vla_adapter", "vla-adapter"]:
        return VLAArchitecture.VLA_ADAPTER

    # Priority 2: Check command-line arguments
    cmd_args = " ".join(sys.argv).lower()
    
    # OpenVLA indicators
    if any(x in cmd_args for x in ["finetune_openvla", "openvla-7b", "openvla/openvla", "use_hf_model"]):
        return VLAArchitecture.OPENVLA
        
    # VLA-Adapter indicators
    if any(x in cmd_args for x in ["finetune.py", "vla-adapter", "vla_adapter", "qwen", "0.5b", "0_5b"]):
        return VLAArchitecture.VLA_ADAPTER

    # Priority 3: Default
    return VLAArchitecture.VLA_ADAPTER

# --- 3. Robot Platform Constants ---

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

def detect_robot_platform() -> str:
    cmd_args = " ".join(sys.argv).lower()
    if "libero" in cmd_args: return "LIBERO"
    if "aloha" in cmd_args: return "ALOHA"
    if "bridge" in cmd_args: return "BRIDGE"
    if "calvin" in cmd_args: return "CALVIN"
    return "LIBERO" # Default if unclear

# --- 4. Execution & Global Assignment ---

# A. Detect Architecture and set Model-specific constants
VLA_ARCH = detect_model_type()
ACTION_TOKEN_BEGIN_IDX = MODEL_CONSTANTS[VLA_ARCH]["ACTION_TOKEN_BEGIN_IDX"]

# B. Detect Robot Platform and set Task-specific constants
ROBOT_PLATFORM = detect_robot_platform()
_platform_map = {
    "LIBERO": LIBERO_CONSTANTS,
    "ALOHA": ALOHA_CONSTANTS,
    "BRIDGE": BRIDGE_CONSTANTS,
    "CALVIN": CALVIN_CONSTANTS
}
_constants = _platform_map[ROBOT_PLATFORM]

if VLA_ARCH == VLAArchitecture.OPENVLA:
    print(f"[Config] Detected OpenVLA architecture. Forcing NUM_ACTIONS_CHUNK = 1 (Single-step prediction).")
    
    # recover the NUM_ACTIONS_CHUNK to 1, since OpenVLA is single-step prediction.
    LIBERO_CONSTANTS["NUM_ACTIONS_CHUNK"] = 1
    CALVIN_CONSTANTS["NUM_ACTIONS_CHUNK"] = 1
    ALOHA_CONSTANTS["NUM_ACTIONS_CHUNK"] = 1  
    BRIDGE_CONSTANTS["NUM_ACTIONS_CHUNK"] = 1
    NUM_TOKENS = 1

elif VLA_ARCH == VLAArchitecture.VLA_ADAPTER:
    STOP_INDEX = 2  # '</s>'
    NUM_TOKENS = 64

# C. Export Global Constants (This is what other modules will import)
NUM_ACTIONS_CHUNK = _constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = _constants["ACTION_DIM"]
PROPRIO_DIM = _constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = _constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# --- 5. Logging (Rank 0 only) ---

def _log_constants():
    try:
        # Check if distributed is initialized; if not, we are likely in a single script or debug mode
        should_log = not dist.is_initialized() or dist.get_rank() == 0
    except (ImportError, AttributeError):
        should_log = True
        
    if should_log:
        print(f"\n[Constants] ===========================================")
        print(f"[Constants] Architecture: {VLA_ARCH.value} (Auto-detected)")
        print(f"[Constants] > ACTION_TOKEN_BEGIN_IDX: {ACTION_TOKEN_BEGIN_IDX}")
        print(f"[Constants] -------------------------------------------")
        print(f"[Constants] Robot Platform: {ROBOT_PLATFORM}")
        print(f"[Constants] > NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}")
        print(f"[Constants] > ACTION_DIM: {ACTION_DIM}")
        print(f"[Constants] > PROPRIO_DIM: {PROPRIO_DIM}")
        print(f"[Constants] > NORM_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
        print(f"[Constants] ===========================================\n")

_log_constants()