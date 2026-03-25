from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ActivationCollector:
    """Registers forward hooks on transformer decoder layers to capture residual stream activations."""

    def __init__(self) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def register(self, model: nn.Module, layer_indices: list[int]) -> None:
        """Register forward hooks on model.model.layers[idx] for each idx in layer_indices."""
        layers = model.model.layers
        for idx in layer_indices:
            key = f"layer_{idx}"

            def make_hook(k: str):
                def hook(module: nn.Module, input: Any, output: Any) -> None:
                    tensor = output[0] if isinstance(output, tuple) else output
                    tensor = tensor.detach().cpu().float()
                    # Squeeze batch dim if present: [1, seq_len, hidden] -> [seq_len, hidden]
                    if tensor.dim() == 3 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    self._activations[k] = tensor

                return hook

            handle = layers[idx].register_forward_hook(make_hook(key))
            self._handles.append(handle)

    def collect(self) -> dict[str, torch.Tensor]:
        """Return the collected activations and clear the internal buffer."""
        result = self._activations
        self._activations = {}
        return result

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_all()
