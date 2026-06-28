"""
Explainability + neurophysiology validation -- the contribution layer.

Input: an importance map shaped like the feature tensor [C, B, W] (from the
linear model's coefficients, or SHAP/Grad-CAM on a deep model -- any method
that scores input elements). We then:

  1. Marginalise it into per-channel, per-band, per-time-window profiles.
  2. Run quantitative agreement tests against established markers:
       - Do FRONTAL channels carry more importance than the rest?
       - Are ALPHA / BETA / THETA the top-weighted bands?
       - Does per-channel ALPHA importance track measured frontal-alpha
         asymmetry across channels?  (the FAA validation)
       - Where in the 15 s epoch does importance concentrate, and does it
         differ between Listen and Recall phases?

These tests turn "here's a heatmap" into "the model's explanation agrees /
disagrees with neurophysiology, significantly, by this measure."
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from config import (
    BAND_NAMES, DASPS_CHANNELS, CH_IDX, FRONTAL_CHANNELS,
    LEFT_FRONTAL, RIGHT_FRONTAL,
)


def flat_importance_to_map(flat_imp, idx, shape):
    """Scatter a per-flat-feature importance back into [C, B, W]."""
    C, B, W = shape
    m = np.zeros((C, B, W), dtype=float)
    for val, (c, b, w) in zip(flat_imp, idx):
        m[c, b, w] = val
    return m


def marginal_profiles(imp_map):
    """[C,B,W] -> dict of per-axis importance profiles (sum-normalised)."""
    per_ch = imp_map.sum(axis=(1, 2))
    per_band = imp_map.sum(axis=(0, 2))
    per_win = imp_map.sum(axis=(0, 1))
    norm = lambda v: v / (v.sum() + 1e-12)
    return dict(channel=norm(per_ch), band=norm(per_band), window=norm(per_win))


def test_frontal_concentration(per_ch):
    """Frontal vs non-frontal channel importance (one-sided Mann-Whitney)."""
    fi = [CH_IDX[c] for c in FRONTAL_CHANNELS]
    oi = [i for i in range(len(DASPS_CHANNELS)) if i not in fi]
    u, p = mannwhitneyu(per_ch[fi], per_ch[oi], alternative="greater")
    return dict(frontal_mean=float(per_ch[fi].mean()),
                other_mean=float(per_ch[oi].mean()),
                p_value=float(p))


def band_ranking(per_band):
    """Bands ordered by importance, with the alpha/beta/theta share."""
    order = np.argsort(per_band)[::-1]
    ranked = [(BAND_NAMES[i], float(per_band[i])) for i in order]
    key = sum(per_band[BAND_NAMES.index(b)] for b in ("alpha", "beta", "theta"))
    return dict(ranked=ranked, alpha_beta_theta_share=float(key))


def faa_alignment(imp_map, EEG, y):
    """Does the model lean on alpha where alpha is actually discriminative?

    Two complementary checks, both tied to frontal alpha asymmetry (FAA):

    (1) Per-channel alignment: across the frontal electrodes, correlate the
        model's ALPHA-band importance with each channel's alpha
        *discriminability* -- |mean log-alpha(anxious) - mean log-alpha(non)|.
        Positive Spearman rho => the model weights alpha on exactly the
        electrodes whose alpha separates the classes (the FAA electrodes).

    (2) Marker reality check: is measured FAA itself different between classes
        here? If FAA doesn't separate the classes in the data, no explanation
        could be expected to recover it -- so we report this alongside.
    """
    from neuromarkers import _bandpower, frontal_alpha_asymmetry
    alpha_imp = imp_map[:, BAND_NAMES.index("alpha"), :].sum(axis=1)  # per channel

    a = np.log(_bandpower(EEG, "alpha") + 1e-10)  # [ep, ch]
    pos, neg = a[y == 1], a[y == 0]
    discrim = np.abs(pos.mean(0) - neg.mean(0))    # per-channel, model-free

    fi = [CH_IDX[c] for c in FRONTAL_CHANNELS]
    if np.ptp(alpha_imp[fi]) == 0 or np.ptp(discrim[fi]) == 0:
        rho, p = float("nan"), float("nan")   # importance is constant/empty
    else:
        rho, p = spearmanr(alpha_imp[fi], discrim[fi])

    faa, _ = frontal_alpha_asymmetry(EEG)
    u, p_faa = mannwhitneyu(faa[y == 1], faa[y == 0], alternative="two-sided")
    return dict(spearman_rho=float(rho), p_value=float(p),
                faa_class_diff=float(faa[y == 1].mean() - faa[y == 0].mean()),
                faa_separates_p=float(p_faa))


def phase_importance(imp_per_epoch, phase):
    """Mean importance magnitude in Listen vs Recall epochs (if provided)."""
    if imp_per_epoch is None:
        return None
    listen = imp_per_epoch[phase == 0].mean()
    recall = imp_per_epoch[phase == 1].mean()
    return dict(listen=float(listen), recall=float(recall))


def run_validation(imp_map, EEG, y, axis_info, verbose=True):
    """Run all neurophysiology-agreement tests, return a report dict."""
    prof = marginal_profiles(imp_map)
    report = dict(
        frontal=test_frontal_concentration(prof["channel"]),
        bands=band_ranking(prof["band"]),
        faa=faa_alignment(imp_map, EEG, y),
        channel_profile=dict(zip(DASPS_CHANNELS, prof["channel"].round(4).tolist())),
        band_profile=dict(zip(BAND_NAMES, prof["band"].round(4).tolist())),
        window_profile=[round(v, 4) for v in prof["window"].tolist()],
        windows=axis_info["windows"],
    )
    if verbose:
        _print_report(report)
    return report


def _print_report(r):
    print("\n=== Explanation vs neurophysiology ===")
    f = r["frontal"]
    print(f"Frontal concentration : frontal={f['frontal_mean']:.4f} "
          f"vs other={f['other_mean']:.4f}  (p={f['p_value']:.4g}, "
          f"{'PASS' if f['p_value'] < 0.05 else 'n.s.'})")
    print("Band ranking          : " +
          ", ".join(f"{b}={v:.3f}" for b, v in r["bands"]["ranked"]))
    print(f"  alpha+beta+theta share = {r['bands']['alpha_beta_theta_share']:.3f}")
    fa = r["faa"]
    print(f"FAA alignment (alpha)  : Spearman rho={fa['spearman_rho']:.3f} "
          f"(p={fa['p_value']:.4g})")
    print(f"  measured FAA class-diff={fa['faa_class_diff']:+.3f} "
          f"(separates classes p={fa['faa_separates_p']:.4g})")
    top_ch = sorted(r["channel_profile"].items(), key=lambda kv: -kv[1])[:5]
    print("Top channels          : " + ", ".join(f"{c}={v:.3f}" for c, v in top_ch))
