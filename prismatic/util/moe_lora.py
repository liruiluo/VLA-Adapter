"""
moe_lora.py

Minimal Mixture-of-Experts (MoE) LoRA implementation for wrapping Linear layers.

This module is intentionally lightweight and self-contained so that it can be
used alongside the existing PEFT-based LoRA path without changing external
dependencies.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoRALinear(nn.Module):
    """
    Wraps a frozen Linear layer with a small MoE-style LoRA adapter.

    - `base` is kept frozen.
    - We instantiate `num_experts` low-rank adapters (A_e, B_e).
    - A token-level router produces a softmax over experts.
    - The final output is: base(x) + sum_e gate_e * (B_e @ A_e @ x).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 4,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"MoELoRALinear expects nn.Linear, got {type(base_linear)}")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self._base_dtype = base_linear.weight.dtype
        self._base_device = base_linear.weight.device

        self.num_experts = int(num_experts)
        self.r = int(r)
        self.scaling = float(lora_alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if self.r <= 0:
            raise ValueError(f"MoE-LoRA requires `r > 0`, got r={self.r}")
        if self.num_experts <= 0:
            raise ValueError(f"MoE-LoRA requires `num_experts > 0`, got num_experts={self.num_experts}")

        self.top_k = None
        if top_k is not None:
            if top_k == 0:
                self.top_k = None
            elif not (0 < top_k < self.num_experts):
                raise ValueError(f"MoE-LoRA requires `0 < top_k < num_experts`, got top_k={top_k}")
            else:
                self.top_k = int(top_k)

        # Expert-specific low-rank matrices
        # A: [E, R, in_features], B: [E, out_features, R]
        self.A = nn.Parameter(
            torch.empty(self.num_experts, self.r, self.in_features, device=self._base_device, dtype=self._base_dtype)
        )
        self.B = nn.Parameter(
            torch.empty(self.num_experts, self.out_features, self.r, device=self._base_device, dtype=self._base_dtype)
        )

        # Simple router over experts: x -> logits[E]
        self.router = nn.Linear(self.in_features, self.num_experts, bias=False)

        self.reset_parameters()

        # Freeze base weights – only train A/B + router
        for p in self.base.parameters():
            p.requires_grad = False

        self.register_buffer("_disabled", torch.tensor(0, dtype=torch.uint8))
        # Ensure dtype/device always matches the wrapped base Linear
        self.to(device=self._base_device, dtype=self._base_dtype)

    def reset_parameters(self) -> None:
        # LoRA-style init: A ~ small random, B ~= 0, router ~= 0
        # Flatten A for initialization
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.A, -bound, bound)
        nn.init.zeros_(self.B)
        nn.init.zeros_(self.router.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback: if disabled, just call base
        if self._disabled.item() == 1:
            return self.base(x)

        base_out = self.base(x)

        # Support inputs of shape [..., in_features]
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)  # [N, D]

        # Router: [N, D] -> [N, E], optionally with top-k sparsification
        logits = self.router(x_flat)  # [N, E]
        if self.top_k is not None:
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            masked = logits.new_full(logits.shape, float("-inf"))
            masked.scatter_(1, topk_idx, topk_vals)
            gate = F.softmax(masked, dim=-1)
        else:
            gate = F.softmax(logits, dim=-1)  # [N, E]

        x_drop = self.dropout(x_flat)  # [N, D]

        # A: [E, R, D], B: [E, O, R]
        # LoRA A:  x_drop [N,D] * A [E,R,D] -> [N,E,R]
        # (einsum: n d, e r d -> n e r)
        lora_A = torch.einsum("nd,erd->ner", x_drop, self.A)
        # LoRA B:  lora_A [N,E,R] * B [E,O,R] -> [N,E,O]
        # (einsum: n e r, e o r -> n e o)
        lora_B = torch.einsum("ner,eor->neo", lora_A, self.B)
        lora_B = lora_B * self.scaling  # [N,E,O]

        # Combine experts with gate: gate [N,E], lora_B [N,E,O] -> [N,O]
        moe_delta = torch.einsum("ne,neo->no", gate, lora_B)

        moe_delta = moe_delta.view(*orig_shape[:-1], self.out_features)
        return base_out + moe_delta


def _should_wrap(name: str, target_modules: Iterable[str]) -> bool:
    """
    Decide whether to wrap a given module name with MoE-LoRA.

    If "all-linear" is present in target_modules, every Linear layer is wrapped.
    Otherwise, we wrap only if any target substring appears in the module name.
    """
    if "all-linear" in target_modules:
        return True
    return any(t in name for t in target_modules)


def apply_moe_lora(
    module: nn.Module,
    num_experts: int,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    top_k: int | None = None,
    target_modules: Iterable[str] = ("all-linear",),
    prefix: str = "",
) -> int:
    """
    Recursively wrap selected Linear submodules of `module` with `MoELoRALinear`.

    Args:
        module: Root module to modify in-place.
        num_experts: Number of experts in the MoE adapter.
        r: LoRA rank per expert.
        lora_alpha: LoRA scaling factor (alpha / r).
        lora_dropout: LoRA dropout probability.
        target_modules: Iterable of substrings; if contains "all-linear", all
            Linear layers are wrapped; otherwise only those whose qualified
            names contain any of the substrings will be wrapped.
        prefix: Internal use; qualified name prefix during recursion.

    Returns:
        The number of `nn.Linear` layers wrapped with `MoELoRALinear`.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        qual_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and _should_wrap(qual_name, target_modules):
            wrapped = MoELoRALinear(
                child,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k,
            )
            setattr(module, name, wrapped)
            replaced += 1
        else:
            replaced += apply_moe_lora(
                child,
                num_experts=num_experts,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                top_k=top_k,
                target_modules=target_modules,
                prefix=qual_name,
            )

    if prefix == "":
        print(f"[MoE-LoRA] Wrapped {replaced} Linear layers with `MoELoRALinear`.")
        if replaced == 0:
            raise ValueError(
                "apply_moe_lora did not wrap any `nn.Linear` layers. "
                "Check `target_modules` (and ensure the model actually contains `nn.Linear` layers)."
            )

    return replaced
