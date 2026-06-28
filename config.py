"""
Central configuration for the DASPS explainable-anxiety pipeline.

Everything that downstream code needs to agree on lives here:
the channel montage, the frequency bands, the frontal-asymmetry pairs,
and the feature/representation hyperparameters. Change values here, not
scattered through the code.
"""

# --- Acquisition constants (fixed by the DASPS recording protocol) ---------
SFREQ = 128            # Emotiv EPOC sampling rate (Hz)
EPOCH_SEC = 15         # length of one DASPS segment (seconds)

# Channel order as recorded by the Emotiv EPOC in DASPS.
# IMPORTANT: confirm this order against the actual files you download.
# If your .mat stores channels in a different order, fix it in data.py,
# NOT here, so every downstream index stays valid.
DASPS_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# --- Frequency bands (Hz) --------------------------------------------------
# NOTE: the DASPS *preprocessed* data is FIR band-pass filtered 4-45 Hz
# (Baghdadi et al., 2019), so there is no genuine sub-4 Hz delta content --
# a "delta" band here would just capture filter roll-off noise. We therefore
# use theta/alpha/beta/gamma. (If you ever switch to the raw .edf, add
# "delta": (1.0, 4.0) back at the front.)
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS.keys())

# --- Frontal alpha asymmetry (FAA) definition ------------------------------
# Standard FAA contrasts homologous left/right frontal sites.
# Convention used here: FAA = ln(power_right) - ln(power_left) in alpha.
# Positive FAA -> relatively greater LEFT alpha (= lower left activity),
# a pattern repeatedly linked to withdrawal/negative affect & anxiety.
FAA_PAIRS = [("F3", "F4"), ("F7", "F8"), ("AF3", "AF4"), ("FC5", "FC6")]
LEFT_FRONTAL = ["F3", "F7", "AF3", "FC5"]
RIGHT_FRONTAL = ["F4", "F8", "AF4", "FC6"]
# Channels we treat as "frontal" when testing whether the model concentrates
# importance frontally (used in the neurophysiology-agreement tests).
FRONTAL_CHANNELS = ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"]

# --- Channel<->index helpers ----------------------------------------------
CH_IDX = {ch: i for i, ch in enumerate(DASPS_CHANNELS)}

# --- Representation hyperparameters (channel x band x time tensor) ----------
# Each 15 s epoch becomes a tensor [n_channels, n_bands, n_windows] of
# log band-power, where the time axis is short sliding sub-windows. This is
# what makes electrode / band / time all first-class explanation axes.
WIN_SEC = 3.0          # sub-window length (seconds)
STEP_SEC = 1.5         # hop between sub-windows (seconds)
LOG_POWER = True       # log-transform band power (stabilises variance)

# --- Task / labelling ------------------------------------------------------
# "binary": anxious vs non-anxious (primary task, recommended headline)
# "fourclass": normal/light/moderate/severe (secondary, imbalanced)
TASK = "binary"
LABEL_SOURCE = "HAMA"  # "HAMA" (recommended primary) or "SAM"

# --- Reproducibility -------------------------------------------------------
SEED = 13
