"""
Validation machinery -- the part that protects you from the single most common
way DASPS results get inflated: subject leakage.

Rules enforced here:
  * Splits are Leave-One-Subject-Out (LOSO): the held-out subject's epochs
    never appear in training.
  * Any statistic that "learns" from data (here: per-feature standardisation)
    is fit on the TRAIN fold only and applied to the test fold. Fitting a
    scaler on all data -- a frequent silent bug -- leaks test distribution
    into training and is forbidden by construction below.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def loso_splits(subj):
    """Yield (train_idx, test_idx, held_out_subject) for each subject."""
    logo = LeaveOneGroupOut()
    dummy = np.zeros(len(subj))
    for tr, te in logo.split(dummy, groups=subj):
        yield tr, te, int(subj[te][0])


def mixed_splits(subj, y, k=5, seed=13):
    """Subject-AGNOSTIC stratified k-fold: segments from the same subject can
    land in both train and test. This is the LEAKY protocol that inflates
    DASPS numbers in the literature -- included precisely so you can show the
    gap against honest LOSO. Not for headline results."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        yield tr, te, -1


def within_subject_splits(subj, y, min_per_class=2):
    """Per-subject leave-one-SITUATION-out, pooled across subjects.

    Each DASPS situation = 2 consecutive segments (recitation + recall) that
    share a SAM label, so we hold out BOTH segments of a situation together.
    Holding out only one segment would leak its near-identical twin (same
    label) into training and inflate the within-subject score. Subjects whose
    own labels are single-class are skipped (cannot classify a constant label).
    """
    idx_all = np.arange(len(subj))
    for s in np.unique(subj):
        sidx = idx_all[subj == s]
        if len(np.unique(y[sidx])) < 2:
            continue
        n_sit = len(sidx) // 2
        for k in range(n_sit):
            te = sidx[2 * k: 2 * k + 2]            # both segments of situation k
            tr = np.concatenate([sidx[:2 * k], sidx[2 * k + 2:]])
            if len(np.unique(y[tr])) < 2:
                continue
            yield tr, te, int(s)


def fit_apply_standardizer(X_train, X_test):
    """Standardise features using TRAIN statistics only.

    X_* are 2-D [n_epochs, n_features]. Returns standardised (train, test).
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def subject_normalize(flat, subj):
    """Per-subject (leak-free) feature z-scoring.

    Each subject's feature vectors are standardised using ONLY that subject's
    own epochs -- no labels, no other subjects -- so it is valid for the
    held-out subject too (it is just per-subject calibration to a personal
    baseline). In cross-subject EEG this typically removes the dominant
    inter-subject offset that otherwise swamps the anxiety signal.
    """
    out = flat.copy()
    for s in np.unique(subj):
        m = subj == s
        mu = flat[m].mean(axis=0, keepdims=True)
        sd = flat[m].std(axis=0, keepdims=True) + 1e-8
        out[m] = (flat[m] - mu) / sd
    return out
