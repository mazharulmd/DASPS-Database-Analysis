"""
Representation: turn raw epochs into the channel x band x time tensor.

For every 15 s epoch we produce a tensor

        T[epoch] : [n_channels, n_bands, n_windows]   of log band-power

built by (1) zero-phase band-pass filtering each canonical band, (2) taking
the analytic (Hilbert) envelope as instantaneous power, and (3) averaging that
power inside short sliding sub-windows along time.

This is the heart of why electrode / band / time are all explainable: every
element of the tensor is indexed by exactly one (channel, band, time-window),
so any attribution method that scores input elements yields a value per axis.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from config import (
    SFREQ, BANDS, BAND_NAMES, WIN_SEC, STEP_SEC, LOG_POWER, DASPS_CHANNELS,
)


def _bandpass(x, lo, hi, fs=SFREQ, order=4):
    """Zero-phase Butterworth band-pass along the last axis."""
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _window_bounds(n_times, fs=SFREQ, win_sec=WIN_SEC, step_sec=STEP_SEC):
    w = int(round(win_sec * fs))
    s = int(round(step_sec * fs))
    starts = list(range(0, n_times - w + 1, s))
    return [(st, st + w) for st in starts]


def epochs_to_tensor(EEG, fs=SFREQ):
    """EEG [n_epochs, n_ch, n_times] -> tensor [n_epochs, n_ch, n_bands, n_win].

    Returns (tensor, axis_info) where axis_info documents what each index means
    -- crucial for reading explanations back out later.
    """
    n_ep, n_ch, n_t = EEG.shape
    bounds = _window_bounds(n_t, fs)
    n_win = len(bounds)
    n_band = len(BANDS)

    out = np.zeros((n_ep, n_ch, n_band, n_win), dtype=np.float32)
    for bi, (lo, hi) in enumerate(BANDS.values()):
        filt = _bandpass(EEG, lo, hi, fs)            # [ep, ch, t]
        power = np.abs(hilbert(filt, axis=-1)) ** 2  # instantaneous power
        for wi, (a, b) in enumerate(bounds):
            out[:, :, bi, wi] = power[:, :, a:b].mean(axis=-1)

    if LOG_POWER:
        out = np.log(out + 1e-10).astype(np.float32)

    axis_info = dict(
        channels=list(DASPS_CHANNELS),
        bands=list(BAND_NAMES),
        windows=[(round(a / fs, 2), round(b / fs, 2)) for a, b in bounds],
        shape="[n_epochs, n_channels, n_bands, n_windows]",
    )
    return out, axis_info


def flatten_tensor(T):
    """[n_ep, C, B, W] -> [n_ep, C*B*W] for classical ML, with an index map."""
    n_ep, C, B, W = T.shape
    flat = T.reshape(n_ep, C * B * W)
    # map flat feature index -> (channel_idx, band_idx, window_idx)
    idx = np.array([(c, b, w) for c in range(C) for b in range(B) for w in range(W)])
    return flat, idx


def epochs_to_connectivity(EEG, fs=SFREQ):
    """Phase-Locking Value (PLV) functional connectivity per band.

    For each frequency band we band-pass filter, take the instantaneous phase
    (Hilbert), and for every pair of electrodes compute
        PLV = | mean_t exp(i * (phase_i - phase_j)) |
    over the 15 s epoch. PLV measures how consistently two regions stay phase-
    synchronised -- a standard functional-connectivity marker, and altered
    fronto-parietal/inter-hemispheric connectivity is repeatedly reported in
    anxiety, which justifies these features.

    Returns (conn [n_epochs, n_bands, n_pairs], axis_info).
    """
    n_ep, n_ch, n_t = EEG.shape
    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]  # 91
    out = np.zeros((n_ep, len(BANDS), len(pairs)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(BANDS.values()):
        filt = _bandpass(EEG, lo, hi, fs)
        phase = np.angle(hilbert(filt, axis=-1))            # [ep, ch, t]
        for pidx, (i, j) in enumerate(pairs):
            dphi = phase[:, i, :] - phase[:, j, :]          # [ep, t]
            out[:, bi, pidx] = np.abs(np.mean(np.exp(1j * dphi), axis=-1))
    axis_info = dict(
        bands=list(BAND_NAMES),
        pairs=[(DASPS_CHANNELS[i], DASPS_CHANNELS[j]) for i, j in pairs],
        shape="[n_epochs, n_bands, n_pairs(91)]",
    )
    return out, axis_info


def epochs_to_extended(EEG, fs=SFREQ):
    """Richer per-epoch features for classical ML (no extra dependencies).

    Per channel we compute:
      * band power (theta/alpha/beta/gamma)  -> 14 x 4 = 56
      * statistics: std, skewness, kurtosis  -> 14 x 3 = 42
      * Hjorth parameters: activity, mobility, complexity -> 14 x 3 = 42
    Total = 140 features. Returns [n_epochs, 140].
    """
    from scipy.stats import skew, kurtosis
    n_ep, n_ch, n_t = EEG.shape

    # band power per channel/band (mean over time), reuse the tensor
    tensor, _ = epochs_to_tensor(EEG, fs)              # [n,14,nbands,nwin]
    bp = tensor.mean(axis=3).reshape(n_ep, -1)         # [n, 14*nbands]

    # statistical features
    std = EEG.std(axis=-1)                             # [n,14]
    sk = skew(EEG, axis=-1)
    ku = kurtosis(EEG, axis=-1)

    # Hjorth parameters
    d1 = np.diff(EEG, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = EEG.var(axis=-1) + 1e-10
    v1 = d1.var(axis=-1) + 1e-10
    v2 = d2.var(axis=-1) + 1e-10
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + 1e-10)

    feats = np.concatenate([bp, std, sk, ku,
                            np.log(activity + 1e-10), mobility, complexity], axis=1)
    return feats.astype(np.float32)
