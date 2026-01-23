"""
openvla.py

PyTorch Module defining OpenVLA and VLAAdapter as lightweight wrappers around a PrismaticVLM; 
defines custom logic around discretizing actions with the ActionTokenizer.

- OpenVLA: Original token-based action prediction
- VLAAdapter: Extended with L1 regression, proprio, multi-image, action chunking
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast # Add

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

try:
    import torch_npu
    USE_NPU = True
except ImportError:
    USE_NPU = False

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    """Original OpenVLA model (token-based action prediction)."""
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    def _process_input_ids(self, input_ids, tokenizer):
        """
        Base class default processing logic (specifically a hack for Llama 2).
        If OpenVLA switches to other backbones in the future, modifications can be made here.
        """
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Llama 2 specific hack: Ensure the prompt ends with a space token
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
            return input_ids
        raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

    @torch.inference_mode()
    def predict_action(
        self, image: Union[Img, List[Img]], instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        # === Modification: Call the specialized processing method ===
        input_ids = self._process_input_ids(input_ids, tokenizer)

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        device = "npu" if USE_NPU else "cuda"
        with torch.autocast(device, dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]


class VLAAdapter(OpenVLA):
    """
    VLA-Adapter (based on Qwen2 or other models).
    """
    def _process_input_ids(self, input_ids, tokenizer):
        # VLAAdapter indeed uses Qwen2, and Qwen2 requires no special processing
        if isinstance(tokenizer, Qwen2TokenizerFast):
            return input_ids
        
        return super()._process_input_ids(input_ids, tokenizer)