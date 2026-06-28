"""
Models.

Two tiers, by design:

  LinearBaseline   -- L2 logistic regression on the flattened channel x band x
                      time tensor. Pure scikit-learn, no GPU. This is your
                      honest WEEK-1 FLOOR under LOSO; if a deep net can't beat
                      it, you have a problem worth knowing about early.

  BandTensorCNN    -- compact PyTorch CNN over the [C, B, W] tensor. Bands act
                      as input channels; convolutions mix electrodes and time
                      windows. Natural target for SHAP / Grad-CAM on the tensor,
                      so explanations come out per (channel, band, window).

  EEGNet           -- the classic compact EEG CNN on the RAW signal [1, C, T],
                      provided for the raw-signal track / comparison.

Torch models are defined only if PyTorch is installed, so this module imports
cleanly on a scipy/sklearn-only machine.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import SEED


class LinearBaseline:
    """L2 logistic regression with a common fit/predict_proba interface.

    Operates on flattened tensors [n_epochs, C*B*W]. Standardisation is handled
    by the caller (train-fold only) -- this class assumes inputs are already
    standardised, keeping the leakage boundary explicit and in one place.
    """

    def __init__(self, C=0.5, multiclass=False):
        self.clf = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=SEED,
        )
        self.multiclass = multiclass

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def coef_importance(self):
        """|coefficients| averaged over classes -> per-flat-feature importance."""
        w = np.abs(self.clf.coef_)
        return w.mean(axis=0)  # [n_features]


# ---------------------------------------------------------------------------
# PyTorch models (optional)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class BandTensorCNN(nn.Module):
        """Compact CNN over the [B, C, W] band-power tensor (B as in-channels)."""

        def __init__(self, n_bands, n_ch, n_win, n_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(n_bands, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ELU(),
                nn.Dropout(0.3),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):  # x: [N, B, C, W]
            z = self.features(x).flatten(1)
            return self.head(z)

    class EEGNet(nn.Module):
        """Minimal EEGNet for raw signal input [N, 1, C, T]."""

        def __init__(self, n_ch, n_times, n_classes=2, F1=8, D=2, kern=64):
            super().__init__()
            F2 = F1 * D
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),  # depthwise
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
                nn.Conv2d(F2, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
            )
            self.head = nn.LazyLinear(n_classes)

        def forward(self, x):  # [N, 1, C, T]
            z = self.block2(self.block1(x))
            return self.head(z.flatten(1))

    TORCH_AVAILABLE = True

except Exception:  # torch not installed
    TORCH_AVAILABLE = False

    class _NoTorch:
        def __init__(self, *a, **k):
            raise ImportError(
                "PyTorch is not installed. `pip install torch` to use the deep "
                "models; the scipy/sklearn LinearBaseline works without it."
            )

    BandTensorCNN = EEGNet = _NoTorch
