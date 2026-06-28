"""
Rigorous evaluation of the chosen model (ExtraTrees on band-power features).

Adds the rigour an examiner expects on top of the model comparison:
  * Honest hyperparameter tuning via NESTED cross-validation (inner loop tunes,
    outer loop estimates) -- so the reported score is not optimistically biased
    by tuning on the test data.
  * Confusion matrix, per-class precision / recall / F1.
  * ROC curve with AUC.
  * Tuned-vs-default comparison, and the most frequently selected settings.

All figures are written to the Figures/ folder.

Usage:
    python run_eval_best.py --data-root . --figdir Figures
"""
from __future__ import annotations
import os, argparse, warnings
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve,
                             precision_recall_fscore_support)

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor
from validation import fit_apply_standardizer

warnings.filterwarnings("ignore")
S = config.SEED
CLASSES = ["non-anxious", "anxious"]
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 16],
    "max_features": ["sqrt", 0.5],
}


def oof_run(flat, y, tuned):
    """One 5-fold outer pass. If tuned, inner GridSearchCV per fold.
    Returns oof_pred, oof_proba, per-fold AUCs, list of chosen params."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=S)
    oof_pred = np.full(len(y), -1)
    oof_proba = np.full(len(y), np.nan)
    fold_aucs, chosen = [], []
    for tr, te in outer.split(flat, y):
        Xtr, Xte = fit_apply_standardizer(flat[tr], flat[te])
        base = ExtraTreesClassifier(class_weight="balanced", random_state=S)
        if tuned:
            gs = GridSearchCV(base, PARAM_GRID, scoring="roc_auc", cv=3, n_jobs=-1)
            gs.fit(Xtr, y[tr])
            clf = gs.best_estimator_
            chosen.append(tuple(sorted(gs.best_params_.items())))
        else:
            clf = base.set_params(n_estimators=300).fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        oof_proba[te] = proba
        oof_pred[te] = (proba >= 0.5).astype(int)
        fold_aucs.append(roc_auc_score(y[te], proba))
    return oof_pred, oof_proba, fold_aucs, chosen


def save_confusion(cm, figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (out-of-fold)", color="#1F3864", fontweight="bold")
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=14, fontweight="bold")
    fig.tight_layout(); p = os.path.join(figdir, "confusion_matrix.png")
    fig.savefig(p, dpi=140); plt.close(); return p


def save_roc(y, proba, auc_val, figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(fpr, tpr, color="#2E75B6", lw=2.2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9aa3ad", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (out-of-fold)", color="#1F3864", fontweight="bold")
    ax.legend(loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = os.path.join(figdir, "roc_curve.png")
    fig.savefig(p, dpi=140); plt.close(); return p


def save_perclass(prec, rec, f1, figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(2); w = 0.25
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(x - w, prec, w, label="Precision", color="#1F3864")
    ax.bar(x, rec, w, label="Recall", color="#2E75B6")
    ax.bar(x + w, f1, w, label="F1", color="#9aa3ad")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1)
    ax.set_title("Per-class metrics", color="#1F3864", fontweight="bold")
    ax.legend(ncol=3, fontsize=8, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = os.path.join(figdir, "perclass_metrics.png")
    fig.savefig(p, dpi=140); plt.close(); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--figdir", default="Figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task="binary", label_source="SAM")
    else:
        print("No --data-root: synthetic demo."); EEG, y, subj, phase, meta = make_synthetic()
    print(f"Label={meta['label_source']}  epochs={len(y)}  class balance={np.bincount(y).tolist()}")

    tensor, _ = epochs_to_tensor(EEG)
    flat = tensor.mean(axis=3).reshape(len(y), -1)
    print(f"Features=band-power ({flat.shape[1]} dims)  Model=ExtraTrees  "
          f"(subject-dependent, nested 5x3 CV)\n")

    # default vs tuned (honest nested CV)
    _, _, def_aucs, _ = oof_run(flat, y, tuned=False)
    pred, proba, aucs, chosen = oof_run(flat, y, tuned=True)

    print("Hyperparameter tuning (nested CV):")
    print(f"  default ExtraTrees : AUC {np.mean(def_aucs):.3f} \u00b1 {np.std(def_aucs):.3f}")
    print(f"  tuned   ExtraTrees : AUC {np.mean(aucs):.3f} \u00b1 {np.std(aucs):.3f}")
    if chosen:
        common = Counter(chosen).most_common(1)[0][0]
        print(f"  most-selected params: {dict(common)}")

    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    pooled_auc = roc_auc_score(y, proba)
    prec, rec, f1c, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    print(f"\nTuned model (out-of-fold pooled):")
    print(f"  AUC={pooled_auc:.3f}  accuracy={acc:.3f}  macro-F1={f1m:.3f}")
    print(f"  {'class':<13}{'precision':<11}{'recall':<9}{'F1':<7}{'support'}")
    for i, cl in enumerate(CLASSES):
        print(f"  {cl:<13}{prec[i]:<11.3f}{rec[i]:<9.3f}{f1c[i]:<7.3f}{sup[i]}")
    print(f"\n  Confusion matrix (rows=true, cols=pred) [non-anxious, anxious]:")
    print(f"    {cm[0].tolist()}\n    {cm[1].tolist()}")

    paths = [save_confusion(cm, args.figdir),
             save_roc(y, proba, pooled_auc, args.figdir),
             save_perclass(prec, rec, f1c, args.figdir)]
    print("\nFigures saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
