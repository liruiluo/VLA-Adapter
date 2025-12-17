"""
moe_lora.py

MoE-LoRA implementation built on top of PEFT's LoRA injection.

Design goal:
- Keep PEFT as the injection mechanism (for broad model support).
- Replace each injected LoRA Linear with a token-level MoE router and per-expert (A,B) LoRA factors.

This mirrors the style in the MoELoRA_Riemannian reference implementation, but is kept minimal:
- No merging/unmerging support for MoE-LoRA.
- No mixed-batch adapter routing support.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear


@dataclass(frozen=True)
class MoELoRAConfig:
    num_experts: int
    top_k: int
    expert_name_prefix: str = "expert"


class MoELoRALinear(PeftLoraLinear):
    """
    PEFT LoRA Linear with a token-level MoE router over experts.

    After `set_moe()`:
    - We create E experts: adapter names `expert_{i}`.
    - Each expert has its own LoRA A/B modules.
    - A router produces gate probabilities per token and we compute:
        y = base(x) + sum_e gate_e(x) * LoRA_e(x)
    """

    moe: Optional[MoELoRAConfig] = None

    def set_moe(self, num_experts: int, top_k: int) -> None:
        if self.merged:
            raise NotImplementedError("MoE-LoRA layers should not be merged; unmerge before converting.")
        if self.disable_adapters:
            raise RuntimeError("Cannot configure MoE-LoRA while adapters are disabled.")

        num_experts = int(num_experts)
        if num_experts <= 0:
            raise ValueError(f"MoE-LoRA requires `num_experts > 0`, got {num_experts}")

        top_k = int(top_k)
        if not (0 < top_k < num_experts):
            raise ValueError(f"MoE-LoRA requires `0 < top_k < num_experts`, got top_k={top_k} num_experts={num_experts}")

        if "default" not in self.lora_A or "default" not in self.lora_B:
            raise RuntimeError("Expected a PEFT-initialized LoRA layer with adapter_name='default'.")

        if getattr(self, "use_dora", {}).get("default", False):
            raise NotImplementedError("MoE-LoRA does not currently support DoRA adapters in this repo.")

        # Duplicate the default LoRA modules into per-expert adapter names.
        base_A: nn.Linear = self.lora_A["default"]
        base_B: nn.Linear = self.lora_B["default"]
        base_dropout: nn.Module = self.lora_dropout["default"]
        base_scaling = self.scaling["default"]

        expert_names = [f"expert_{i}" for i in range(num_experts)]
        lora_A = nn.ModuleDict({name: copy.deepcopy(base_A) for name in expert_names})
        lora_B = nn.ModuleDict({name: copy.deepcopy(base_B) for name in expert_names})
        lora_dropout = nn.ModuleDict({name: copy.deepcopy(base_dropout) for name in expert_names})

        # Replace PEFT internal adapter maps.
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = {name: base_scaling for name in expert_names}
        self.use_dora = {name: False for name in expert_names}
        self.set_adapter(expert_names)

        # Token-level router: x -> logits over experts
        in_features = base_A.weight.shape[1]
        dtype = base_A.weight.dtype
        device = base_A.weight.device
        self.gate = nn.Linear(in_features, num_experts, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.gate.weight)

        # LoRA-style initialization per expert (A: kaiming, B: zeros)
        for name in expert_names:
            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[name].weight)

        self.moe = MoELoRAConfig(num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.moe is None:
            return super().forward(x, *args, **kwargs)

        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError("MoE-LoRA does not support PEFT mixed-batch forwarding in this repo.")
        if self.merged:
            raise NotImplementedError("MoE-LoRA layers should not be merged; some errors may happen.")

        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        # Compute per-token gate probabilities.
        #
        # IMPORTANT: We do "top-k then softmax" (masked softmax on logits), not "softmax then top-k".
        gate_logits = self.gate(x.to(self.gate.weight.dtype)).to(torch_result_dtype)  # [..., E]
        if self.moe is None:
            raise RuntimeError("MoE-LoRA misconfigured: missing `self.moe`.")
        if not (0 < self.moe.top_k < self.moe.num_experts):
            raise RuntimeError(
                f"MoE-LoRA requires `0 < top_k < num_experts`, got top_k={self.moe.top_k} num_experts={self.moe.num_experts}"
            )
        topk_vals, topk_idx = torch.topk(gate_logits, self.moe.top_k, dim=-1)
        masked = gate_logits.new_full(gate_logits.shape, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        gate_probs = F.softmax(masked, dim=-1)

        # Run experts (dense loop; acceptable for small num_experts)
        # NOTE: assume x is at least 2D (batch x seq x dim) for gate indexing
        for expert_id, expert_name in enumerate(self.active_adapters):
            if expert_name not in self.lora_A:
                continue
            lora_A = self.lora_A[expert_name]
            lora_B = self.lora_B[expert_name]
            dropout = self.lora_dropout[expert_name]
            scaling = self.scaling[expert_name]

            x_cast = x.to(lora_A.weight.dtype)
            delta = lora_B(lora_A(dropout(x_cast))) * scaling  # [..., out]
            weight = gate_probs[..., expert_id].unsqueeze(-1)  # [..., 1]
            result = result + weight * delta.to(torch_result_dtype)

        return result.to(torch_result_dtype)


def apply_moe_lora(
    model: nn.Module,
    *,
    num_experts: int,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    top_k: int,
    target_modules: Union[str, Sequence[str]] = "all-linear",
) -> nn.Module:
    """
    Apply PEFT LoRA to `model` and convert injected LoRA Linear layers into MoE-LoRA layers.

    Returns a PEFT-wrapped model (PeftModel) with MoE-LoRA layers.
    """
    normalized_target_modules: Union[str, Sequence[str]] = target_modules
    if not isinstance(target_modules, str):
        # Be forgiving: many call sites pass ["all-linear"] or include it among substrings.
        if "all-linear" in target_modules:
            normalized_target_modules = "all-linear"

    lora_config = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=normalized_target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    converted = 0
    for _, module in peft_model.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer) and isinstance(module, PeftLoraLinear):
            module.__class__ = MoELoRALinear
            module.set_moe(num_experts=num_experts, top_k=top_k)
            converted += 1

    print(f"[MoE-LoRA/PEFT] Converted {converted} LoRA Linear layers to `MoELoRALinear`.")
    if converted == 0:
        raise ValueError(
            "apply_moe_lora did not convert any PEFT LoRA Linear layers. "
            "Check `target_modules` and ensure LoRA injection succeeded."
        )

    return peft_model
