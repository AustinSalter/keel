from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Model identifiers
TRINITY_MINI_ID = "arcee-ai/Trinity-Mini-Base"
QWEN_FALLBACK_ID = "Qwen/Qwen2.5-7B"

# Layer indices (0-indexed) for activation capture
# Spec says {8, 16, 24, 32} (1-indexed) → {7, 15, 23, 31} (0-indexed)
TRINITY_LAYERS = [7, 15, 23, 31]
TRINITY_FALLBACK_LAYERS = [15, 31]  # Memory-constrained fallback

# Qwen 2.5 7B has 28 layers — evenly spaced equivalent
QWEN_LAYERS = [6, 13, 20, 27]
QWEN_FALLBACK_LAYERS = [13, 27]

# Layer indices derived from Sprint 0 layer sweep
# Layers 7, 11: optimal zone (EVR 0.87+, SNR 12-18x)
# Layer 24: post-noise-wall recovery zone
SWEEP_LAYERS = [7, 11, 24]

# Supabase config (Phello)
SUPABASE_URL = "https://qfyufzqfjxfiqehnveck.supabase.co"
PHELLO_USER_ID = "d09855bd-2290-4d52-bb04-13517eef5b81"

# PCA component counts to test
PCA_COMPONENTS = [5, 10, 20]


@dataclass
class CaptureConfig:
    """Configuration for an activation capture run."""

    model_id: str = TRINITY_MINI_ID
    layer_indices: list[int] = field(default_factory=lambda: list(SWEEP_LAYERS))
    dtype: torch.dtype = torch.bfloat16
    max_seq_length: int = 4096
    pca_components: list[int] = field(default_factory=lambda: list(PCA_COMPONENTS))
    device_map: str = "auto"
    trust_remote_code: bool = True
