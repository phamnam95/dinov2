import math
from typing import Iterable, Optional, Sequence

import torch
from torch import nn, Tensor


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Base linear params (frozen)
        self.weight = nn.Parameter(base.weight.detach().clone())
        self.weight.requires_grad_(False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone())
            self.bias.requires_grad_(False)
        else:
            self.bias = None

        # LoRA params
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self) -> None:
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.lora_B is not None:
            nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = nn.functional.linear(self.lora_dropout(x), self.lora_A.T)
            lora_out = nn.functional.linear(lora_out, self.lora_B.T)
            result = result + self.scaling * lora_out
        return result


def _replace_child(module: nn.Module, child_name: str, new_child: nn.Module) -> None:
    parts = child_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_child)


def apply_lora_to_vit(
    model: nn.Module,
    *,
    target_substrings: Sequence[str] = ("qkv", "proj", "fc1", "fc2"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Wrap selected Linear layers with LoRA adapters.
    target_substrings selects modules by name suffix (e.g., 'qkv','proj','fc1','fc2').
    """
    named_modules = dict(model.named_modules())
    # Collect linear module names to avoid modifying while iterating
    linear_names = []
    for name, mod in named_modules.items():
        if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in target_substrings):
            linear_names.append(name)

    for name in linear_names:
        base = named_modules[name]
        lora = LoRALinear(base, r=r, lora_alpha=alpha, lora_dropout=dropout)
        _replace_child(model, name, lora)

    return model
