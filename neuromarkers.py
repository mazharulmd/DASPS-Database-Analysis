"""
Classical neural markers, computed INDEPENDENTLY of the deep model.

These are the ground truth your explanations get validated against. The key
one is frontal alpha asymmetry (FAA); we also expose per-channel alpha power
and frontal theta/beta, since your literature repeatedly flags frontal
theta-beta in anxiety.

Computing these here -- not inside the model -- is the whole point: the model
never sees "FAA". If its explanations independently rank frontal alpha and
beta highly and its alpha importance tracks measured FAA, that agreement is
evidence the model learned genuine neurophysiology rather than an artefact.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from config import SFREQ, BANDS, CH_IDX, FAA_PAIRS, FRONTAL_CHANNELS


def _bandpower(EEG, band, fs=SFREQ):
    """Mean band power per epoch/channel -> [n_epochs, n_channels]."""
    lo, hi = BANDS[band]
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band")
    filt = filtfilt(b, a, EEG, axis=-1)
    return (np.abs(hilbert(filt, axis=-1)) ** 2).mean(axis=-1)  # [ep, ch]


def alpha_power_per_channel(EEG):
    """Log alpha power, [n_epochs, n_channels]."""
    return np.log(_bandpower(EEG, "alpha") + 1e-10)


def frontal_alpha_asymmetry(EEG):
    """FAA per epoch = mean over pairs of ln(P_right) - ln(P_left) in alpha.

    Returns (faa [n_epochs], per_pair [n_epochs, n_pairs]).
    Positive => relatively greater left alpha (lower left activity).
    """
    a = _bandpower(EEG, "alpha")  # [ep, ch]
    cols = []
    for l, r in FAA_PAIRS:
        cols.append(np.log(a[:, CH_IDX[r]] + 1e-10) - np.log(a[:, CH_IDX[l]] + 1e-10))
    per_pair = np.stack(cols, axis=1)
    return per_pair.mean(axis=1), per_pair


def frontal_band_power(EEG, band):
    """Mean log power over frontal channels for a band -> [n_epochs]."""
    bp = _bandpower(EEG, band)
    fi = [CH_IDX[c] for c in FRONTAL_CHANNELS]
    return np.log(bp[:, fi].mean(axis=1) + 1e-10)


def marker_table(EEG):
    """Convenience: dict of the standard markers for an epoch set."""
    faa, _ = frontal_alpha_asymmetry(EEG)
    return dict(
        FAA=faa,
        frontal_theta=frontal_band_power(EEG, "theta"),
        frontal_beta=frontal_band_power(EEG, "beta"),
        frontal_alpha=frontal_band_power(EEG, "alpha"),
    )
