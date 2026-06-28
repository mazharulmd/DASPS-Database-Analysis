"""
Accuracy-improvement experiment: extended features + ensemble.

Compares, under the honest subject-dependent protocol (repeated 5-fold):
  * features:  band power (56)   vs   extended (140: + statistics + Hjorth)
  * models:    best singles      vs   a soft-voting ENSEMBLE of them
so you can see directly whether either lever raises AUC over the 0.69 baseline.

Usage:
    python run_ensemble.py --data-root . --repeats 10
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor, epochs_to_extended
from validation import mixed_splits

warnings.filterwarnings("ignore")
S = config.SEED


def factories():
    et = lambda: ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=S)
    rf = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=S)
    sv = lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=S)
    lr = lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    singles = {
        "ExtraTrees":   lambda: make_pipeline(StandardScaler(), et()),
        "RandomForest": lambda: make_pipeline(StandardScaler(), rf()),
        "SVM-RBF":      lambda: make_pipeline(StandardScaler(), sv()),
        "LogReg":       lambda: make_pipeline(StandardScaler(), lr()),
    }
    ens = lambda: make_pipeline(StandardScaler(), VotingClassifier(
        estimators=[("et", et()), ("rf", rf()), ("sv", sv()), ("lr", lr())],
        voting="soft"))
    singles["Ensemble(soft)"] = ens
    return singles


def evaluate(make_est, X, y, subj, repeats):
    aucs, accs, f1s = [], [], []
    for r in range(repeats):
        oofp = np.full(len(y), np.nan); oofd = np.full(len(y), -1)
        for tr, te, _ in mixed_splits(subj, y, k=5, seed=S + r):
            est = make_est().fit(X[tr], y[tr])
            oofp[te] = est.predict_proba(X[te])[:, 1]
            oofd[te] = est.predict(X[te])
        aucs.append(roc_auc_score(y, oofp))
        accs.append(accuracy_score(y, oofd))
        f1s.append(f1_score(y, oofd, average="macro"))
    return np.mean(aucs), np.std(aucs), np.mean(accs), np.mean(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task="binary", label_source="SAM")
    else:
        print("No --data-root: synthetic demo."); EEG, y, subj, phase, meta = make_synthetic()
    print(f"Label={meta['label_source']}  epochs={len(y)}  class balance={np.bincount(y).tolist()}")

    tensor, _ = epochs_to_tensor(EEG)
    feats = {
        "band power (56)": tensor.mean(axis=3).reshape(len(y), -1),
        "extended (140)":  epochs_to_extended(EEG),
    }
    models = factories()
    print(f"Subject-dependent, {args.repeats} repeats of 5-fold. AUC = mean\u00b1std.\n")

    for fname, X in feats.items():
        print(f"--- Features: {fname} ---")
        print(f"  {'Model':<16}{'AUC':<14}{'Accuracy':<10}{'macro-F1'}")
        rows = []
        for mname, mk in models.items():
            au, sd, ac, f1 = evaluate(mk, X, y, subj, args.repeats)
            rows.append((mname, au, sd, ac, f1))
        for mname, au, sd, ac, f1 in rows:
            tag = "  <-- ensemble" if "Ensemble" in mname else ""
            print(f"  {mname:<16}{au:.3f}\u00b1{sd:.2f}    {ac:.3f}     {f1:.3f}{tag}")
        print()

    print("Read: compare the best AUC here against your band-power baseline (~0.69). "
          "If extended features or the ensemble are clearly higher AND stable (small std), "
          "adopt them; if not, keep the simpler setup and report this honestly.")


if __name__ == "__main__":
    main()
