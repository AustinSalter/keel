from __future__ import annotations

import torch
import torch.nn as nn

from substrate.hooks import ActivationCollector


# ---------------------------------------------------------------------------
# Tiny synthetic model that mimics the HuggingFace decoder structure:
#   model.model.layers  — an nn.ModuleList of decoder layers
# ---------------------------------------------------------------------------

class FakeDecoderLayer(nn.Module):
    """A minimal decoder layer: linear projection over the hidden dim."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FakeInner(nn.Module):
    """Mimics model.model with a .layers attribute."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FakeModel(nn.Module):
    """Top-level model with .model attribute, mimicking HuggingFace CausalLM."""

    def __init__(self, hidden_size: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.model = FakeInner(hidden_size, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TupleOutputLayer(nn.Module):
    """A decoder layer that returns a tuple (tensor, extra_info), like real HF layers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        return self.proj(x), "extra"


class FakeModelWithTupleOutput(nn.Module):
    """Model whose layers return tuples."""

    def __init__(self, hidden_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()

        class Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(
                    [TupleOutputLayer(hidden_size) for _ in range(num_layers)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x, _ = layer(x)
                return x

        self.model = Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

HIDDEN = 16
SEQ_LEN = 8
BATCH = 1


def make_input(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, SEQ_LEN, HIDDEN)


def test_register_and_collect() -> None:
    """Hooks capture activations with correct keys and shapes."""
    model = FakeModel(hidden_size=HIDDEN, num_layers=4)
    collector = ActivationCollector()
    collector.register(model, layer_indices=[0, 2])

    model(make_input())

    activations = collector.collect()

    assert set(activations.keys()) == {"layer_0", "layer_2"}
    for key in ("layer_0", "layer_2"):
        tensor = activations[key]
        assert tensor.shape == (SEQ_LEN, HIDDEN), (
            f"{key}: expected ({SEQ_LEN}, {HIDDEN}), got {tensor.shape}"
        )

    collector.remove_all()


def test_context_manager_removes_hooks() -> None:
    """Using the context manager removes all hooks on exit."""
    model = FakeModel(hidden_size=HIDDEN, num_layers=4)

    with ActivationCollector() as collector:
        collector.register(model, layer_indices=[1, 3])
        assert len(collector._handles) == 2

    # After exiting, handles list is cleared and hooks are gone.
    assert len(collector._handles) == 0

    # Confirm hooks are actually removed: a forward pass produces no new activations.
    model(make_input())
    assert collector._activations == {}


def test_tuple_output_handling() -> None:
    """When a layer returns a tuple, collector takes output[0]."""
    model = FakeModelWithTupleOutput(hidden_size=HIDDEN, num_layers=2)
    collector = ActivationCollector()
    collector.register(model, layer_indices=[0, 1])

    model(make_input())

    activations = collector.collect()
    assert set(activations.keys()) == {"layer_0", "layer_1"}
    for tensor in activations.values():
        assert tensor.shape == (SEQ_LEN, HIDDEN)

    collector.remove_all()


def test_collect_clears_buffer() -> None:
    """Calling collect() twice: first call returns data, second returns empty dict."""
    model = FakeModel(hidden_size=HIDDEN, num_layers=2)
    collector = ActivationCollector()
    collector.register(model, layer_indices=[0])

    model(make_input())

    first = collector.collect()
    assert "layer_0" in first

    second = collector.collect()
    assert second == {}

    collector.remove_all()


def test_detach_cpu_float() -> None:
    """Captured tensors are on CPU and in float32, regardless of model dtype."""
    model = FakeModel(hidden_size=HIDDEN, num_layers=2)
    # Run model in float16 to confirm the hook always upcasts to float32.
    model = model.to(torch.float16)

    collector = ActivationCollector()
    collector.register(model, layer_indices=[0])

    model(make_input().to(torch.float16))

    activations = collector.collect()
    tensor = activations["layer_0"]

    assert tensor.device.type == "cpu", f"Expected CPU, got {tensor.device}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert not tensor.requires_grad

    collector.remove_all()
