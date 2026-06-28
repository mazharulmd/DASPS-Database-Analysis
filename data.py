"""
Data layer for the real DASPS dataset (as distributed on Kaggle).

Confirmed structure of THIS download:
  DASPS_Database/
    Preprocessed data .mat/SNNpreprocessed.mat   # 23 files, MATLAB v7.3 (HDF5)
        variable 'data' : shape (12, 1920, 14) = (segments, times, channels)
        12 segments = 6 situations x 2 phases (recitation, recall), 15 s @128Hz
    Raw data .edf/ , Raw data.mat/                # not used here
    participant_rating_public.xlsx               # labels: HAM-A + SAM

Normalised in-memory format used by the whole pipeline:
    EEG   : float32 [n_epochs, n_channels(14), n_times(1920)]
    y     : int     [n_epochs]
    subj  : int     [n_epochs]   (subject id, for LOSO)
    phase : int     [n_epochs]   (0 = recitation/listen, 1 = recall  -- assumed)
    meta  : dict

`load_dasps` accepts EITHER the extracted DASPS_Database folder OR the .zip
file directly -- no manual unzip needed.

Labels: default is HAM-A severity per subject (the file gives the severity tag
explicitly, so no guessing). Binary = {normal,light}->0 vs {moderate,severe}->1.
SAM per-situation labels are also parsed and returned in meta for later use.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
import numpy as np

from config import (
    SFREQ, EPOCH_SEC, DASPS_CHANNELS, CH_IDX,
    LEFT_FRONTAL, RIGHT_FRONTAL, SEED,
)

N_CH = len(DASPS_CHANNELS)
N_TIMES = int(SFREQ * EPOCH_SEC)
SEVERITY = {"normal": 0, "light": 1, "moderate": 2, "severe": 3}


def _sam_label(val, aro, task="binary"):
    """Map a situation's SAM (valence, arousal) -> anxiety level.

    Thresholds follow the published DASPS scheme (Baghdadi et al. 2019;
    Asghar et al. 2022) along the low-valence / high-arousal diagonal,
    evaluated as a cascade (severe first):
        severe  : valence <= 2 and arousal >= 7
        moderate: valence <= 4 and arousal >= 6
        light   : valence <= 5 and arousal >= 5
        normal  : otherwise (positive valence / low arousal)
    4-class -> 0..3 ; binary -> {normal,light}=0 vs {moderate,severe}=1.
    Returns None if valence/arousal are missing.
    """
    if val is None or aro is None:
        return None
    v, a = float(val), float(aro)
    if v <= 2 and a >= 7:
        lvl = 3
    elif v <= 4 and a >= 6:
        lvl = 2
    elif v <= 5 and a >= 5:
        lvl = 1
    else:
        lvl = 0
    return lvl if task == "fourclass" else int(lvl >= 2)


# ===========================================================================
# Low-level readers
# ===========================================================================
def _read_mat_bytes(raw: bytes):
    """Read a MATLAB v7.3 .mat (given as bytes) -> array [seg, time, ch]."""
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as f:
        arr = np.asarray(f["data"][()], dtype=np.float32)
    # h5py gives (12, 1920, 14); orient to [seg, time, ch] by axis lengths.
    ch_axis = next(i for i, s in enumerate(arr.shape) if s == N_CH)
    time_axis = int(np.argmax(arr.shape))
    seg_axis = ({0, 1, 2} - {ch_axis, time_axis}).pop()
    return np.transpose(arr, (seg_axis, time_axis, ch_axis))


def _norm_sev(tag):
    """'25:moderate' / ' 29:zsevere' -> ('severe', 29)."""
    num, _, word = str(tag).partition(":")
    word = word.strip().lower()
    word = {"zsevere": "severe"}.get(word, word)
    try:
        val = int(re.sub(r"[^0-9]", "", num))
    except ValueError:
        val = None
    return word, val


def _read_labels_xlsx(raw: bytes):
    """Parse participant_rating_public.xlsx -> per-subject label dict.

    Returns {subject_int: {sev4, binary, h1, h2, sam:[(val,aro)x6]}}.
    """
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]  # drop header

    out, cur, h1, h2, sam = {}, None, None, None, []
    for r in rows:
        pid, sit, val, aro, ham1, ham2 = r[1], r[2], r[3], r[4], r[5], r[6]
        if pid:  # new subject block starts
            if cur is not None:
                out[cur] = dict(h1=h1, h2=h2, sam=sam)
            cur = int(re.sub(r"[^0-9]", "", str(pid)))
            h1 = h2 = None
            sam = []
        if ham1 not in (None, ""):
            h1 = ham1
        if ham2 not in (None, ""):
            h2 = ham2
        if sit is not None:
            sam.append((val, aro))
    if cur is not None:
        out[cur] = dict(h1=h1, h2=h2, sam=sam)

    # finalise: derive severity class + binary from the chosen Hamilton measure
    for sid, d in out.items():
        for key in ("h1", "h2"):
            word, num = _norm_sev(d[key]) if d[key] is not None else (None, None)
            d[f"{key}_sev"] = word
            d[f"{key}_num"] = num
    return out


# ===========================================================================
# Public loader
# ===========================================================================
def _iter_members(path):
    """Yield (subject_int, mat_bytes) and return label bytes, from zip or dir."""
    pre_re = re.compile(r"S(\d+)preprocessed\.mat$", re.IGNORECASE)
    lab_name = "participant_rating_public.xlsx"

    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        label_bytes = None
        subs = []
        for n in zf.namelist():
            base = os.path.basename(n)
            m = pre_re.search(base)
            if m and "Preprocessed" in n:
                subs.append((int(m.group(1)), n))
            elif base == lab_name:
                label_bytes = zf.read(n)
        subs.sort()
        for sid, n in subs:
            yield ("mat", sid, zf.read(n))
        yield ("labels", None, label_bytes)
    else:
        label_bytes = None
        subs = []
        for root, _, files in os.walk(path):
            for fn in files:
                m = pre_re.search(fn)
                full = os.path.join(root, fn)
                if m and "Preprocessed" in root:
                    subs.append((int(m.group(1)), full))
                elif fn == lab_name:
                    label_bytes = open(full, "rb").read()
        subs.sort()
        for sid, full in subs:
            yield ("mat", sid, open(full, "rb").read())
        yield ("labels", None, label_bytes)


def load_dasps(root, task="binary", label_source="HAMA", hamilton="h1"):
    """Load real DASPS into the normalised format.

    root         : path to DASPS_Database folder OR the .zip
    task         : 'binary' or 'fourclass'
    label_source : 'HAMA' (per-subject severity) or 'SAM' (per-situation,
                   the more learnable, standard DASPS target)
    hamilton     : 'h1' or 'h2' (only used when label_source='HAMA')

    SAM mapping: the 12 preprocessed segments are 6 situations x 2 windows,
    so situation index = segment // 2 (situation-major ordering -- documented
    assumption; verify against the DASPS segmentation script if unsure).
    """
    mats, label_bytes = [], None
    for kind, sid, raw in _iter_members(root):
        if kind == "mat":
            mats.append((sid, raw))
        else:
            label_bytes = raw
    if not mats:
        raise FileNotFoundError(f"No SNNpreprocessed.mat found under {root}")
    if label_bytes is None:
        raise FileNotFoundError("participant_rating_public.xlsx not found")

    labels = _read_labels_xlsx(label_bytes)

    EEG, y, subj, phase = [], [], [], []
    sam_table = {}
    for sid, raw in mats:
        arr = _read_mat_bytes(raw)                 # [seg, time, ch]
        n_seg = arr.shape[0]
        if arr.shape[1] != N_TIMES:
            arr = _fix_time_length(arr)
        arr = np.transpose(arr, (0, 2, 1))         # -> [seg, ch, time]

        ld = labels[sid]
        sam_table[sid] = ld["sam"]

        if label_source.upper() == "SAM":
            seg_labels = []
            for s in range(n_seg):
                sit = s // 2                       # situation index
                val, aro = (ld["sam"][sit] if sit < len(ld["sam"]) else (None, None))
                lab = _sam_label(val, aro, task)
                seg_labels.append(0 if lab is None else lab)
            EEG.append(arr); y += seg_labels
        else:  # HAM-A per subject
            sev_word = ld[f"{hamilton}_sev"]
            sev4 = SEVERITY.get(sev_word, 0)
            label = sev4 if task == "fourclass" else int(sev4 >= 2)
            EEG.append(arr); y += [label] * n_seg

        subj += [sid] * n_seg
        phase += [s % 2 for s in range(n_seg)]

    EEG = np.concatenate(EEG, 0).astype(np.float32)
    meta = dict(ch_names=DASPS_CHANNELS, sfreq=SFREQ, task=task,
                label_source=(f"SAM" if label_source.upper() == "SAM"
                              else f"HAMA:{hamilton}"),
                n_subjects=len(mats), sam=sam_table)
    return EEG, np.asarray(y), np.asarray(subj), np.asarray(phase), meta


def _fix_time_length(arr):
    """Truncate or zero-pad time axis (axis=1) to N_TIMES."""
    t = arr.shape[1]
    if t > N_TIMES:
        return arr[:, :N_TIMES, :]
    pad = np.zeros((arr.shape[0], N_TIMES - t, arr.shape[2]), arr.dtype)
    return np.concatenate([arr, pad], axis=1)


# ===========================================================================
# Synthetic generator (planted ground-truth neurophysiology) -- unchanged
# ===========================================================================
def make_synthetic(n_subjects=23, segs_per_subject=12, effect=1.0, seed=SEED):
    """Fake DASPS-like data with a KNOWN frontal alpha/beta anxiety signature,
    for verifying the pipeline before trusting real results."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIMES) / SFREQ

    def osc(freq, amp, n):
        ph = rng.uniform(0, 2 * np.pi, size=(n, 1))
        return amp * np.sin(2 * np.pi * freq * t[None, :] + ph)

    EEG, y, subj, phase = [], [], [], []
    li = [CH_IDX[c] for c in LEFT_FRONTAL]
    ri = [CH_IDX[c] for c in RIGHT_FRONTAL]
    front = sorted(set(li + ri))

    for sid in range(n_subjects):
        subj_alpha = rng.normal(1.0, 0.25)
        subj_bias = rng.normal(0, 0.4)
        for seg in range(segs_per_subject):
            label = int(seg >= segs_per_subject // 2)
            x = rng.normal(0, 1.0, size=(N_CH, N_TIMES)).astype(np.float32)
            for ch in range(N_CH):
                x[ch] += osc(2.5, 1.2, 1)[0]
                x[ch] += osc(6.0, 1.0, 1)[0]
                x[ch] += osc(10.0, subj_alpha * 1.4, 1)[0]
                x[ch] += osc(20.0, 0.7, 1)[0]
                x[ch] += osc(38.0, 0.3, 1)[0]
            if label == 1:
                for ch in li:
                    x[ch] -= osc(10.0, effect * 0.9, 1)[0]
                for ch in front:
                    x[ch] += osc(20.0, effect * 0.9, 1)[0]
            x += subj_bias
            EEG.append(x); y.append(label); subj.append(sid); phase.append(seg % 2)

    EEG = np.stack(EEG).astype(np.float32)
    meta = dict(ch_names=DASPS_CHANNELS, sfreq=SFREQ, task="binary",
                label_source="SYNTH", n_subjects=n_subjects, synthetic=True)
    return EEG, np.asarray(y), np.asarray(subj), np.asarray(phase), meta
