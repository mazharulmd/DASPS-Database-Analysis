"""
Explainability for the traditional-ML model (the thesis core, classical-ML stage).

Takes the chosen band-power model (default ExtraTrees), computes feature
importance per (electrode, band) -- via SHAP if available, else tree impurity
importance, plus model-agnostic permutation importance -- aggregated across
cross-validation folds, then validates the explanation against neurophysiology
(frontal concentration, band ranking, frontal alpha asymmetry).

Usage:
    python run_xai_ml.py --data-root . --model ExtraTrees
    python run_xai_ml.py --data-root . --model RandomForest --save-fig

This answers the thesis question: WHICH electrodes and bands drive the
anxiety decision, and do they agree with established markers (FAA)?
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor
from validation import fit_apply_standardizer
from xai import run_validation
from run_ml import make_models
from config import BAND_NAMES, DASPS_CHANNELS

warnings.filterwarnings("ignore")
S = config.SEED
N_CH, N_BAND = len(DASPS_CHANNELS), len(BAND_NAMES)


def shap_importance(model, X):
    """Mean |SHAP| per feature for a tree model; None if shap unavailable."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        return np.abs(sv).mean(axis=0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--model", default="ExtraTrees")
    ap.add_argument("--save-fig", action="store_true", help="save a channel x band heatmap PNG")
    args = ap.parse_args()

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task="binary", label_source="SAM")
    else:
        print("No --data-root: synthetic demo."); EEG, y, subj, phase, meta = make_synthetic()
    print(f"Label={meta['label_source']}  epochs={len(y)}  model={args.model}")

    tensor, _ = epochs_to_tensor(EEG)
    flat = tensor.mean(axis=3).reshape(len(y), -1)             # [n, 56] (ch-major, band-fast)

    shap_imp = np.zeros(flat.shape[1])
    tree_imp = np.zeros(flat.shape[1])
    perm_imp = np.zeros(flat.shape[1])
    shap_ok = True
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=S)
    for tr, te in skf.split(flat, y):
        Xtr, Xte = fit_apply_standardizer(flat[tr], flat[te])
        clf = make_models()[args.model].fit(Xtr, y[tr])
        if hasattr(clf, "feature_importances_"):
            tree_imp += clf.feature_importances_
        si = shap_importance(clf, Xte)
        if si is None:
            shap_ok = False
        else:
            shap_imp += si
        pi = permutation_importance(clf, Xte, y[te], n_repeats=10, random_state=S)
        perm_imp += np.clip(pi.importances_mean, 0, None)

    # choose explanation source: SHAP > permutation
    if shap_ok and shap_imp.sum() > 0:
        imp, src = shap_imp, "SHAP"
    else:
        imp, src = perm_imp, "permutation importance"
    print(f"Explanation source: {src} (aggregated over 5 folds)\n")

    imp_map = imp.reshape(N_CH, N_BAND)[:, :, None]            # [14, 4, 1]
    axis_info = dict(channels=list(DASPS_CHANNELS), bands=list(BAND_NAMES),
                     windows=[(0.0, 15.0)], shape="[channels, bands, 1]")
    run_validation(imp_map, EEG, y, axis_info, verbose=True)

    # readable extras
    per_band = imp_map[:, :, 0].sum(0); per_band /= per_band.sum() + 1e-12
    per_ch = imp_map[:, :, 0].sum(1); per_ch /= per_ch.sum() + 1e-12
    print("\nPer-band importance share : " +
          ", ".join(f"{b}={v:.2f}" for b, v in zip(BAND_NAMES, per_band)))
    top = np.argsort(per_ch)[::-1][:6]
    print("Top electrodes            : " +
          ", ".join(f"{DASPS_CHANNELS[i]}={per_ch[i]:.2f}" for i in top))

    if args.save_fig:
        try:
            import os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs("Figures", exist_ok=True)
            M = imp_map[:, :, 0]
            fig, ax = plt.subplots(figsize=(4, 7))
            im = ax.imshow(M, aspect="auto", cmap="viridis")
            ax.set_xticks(range(N_BAND)); ax.set_xticklabels(BAND_NAMES)
            ax.set_yticks(range(N_CH)); ax.set_yticklabels(DASPS_CHANNELS)
            ax.set_title(f"{args.model} importance\n({src})")
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.tight_layout()
            path = os.path.join("Figures", "xai_importance_heatmap.png")
            fig.savefig(path, dpi=130)
            print(f"\nSaved figure: {path}")
        except Exception as e:
            print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
