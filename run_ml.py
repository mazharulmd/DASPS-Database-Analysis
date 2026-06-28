"""
Traditional-ML comparison for DASPS (supervisor-directed first stage).

Per Dr. Ferdowsi: SAM labels, within-subject analysis, preprocessed data,
a RANGE of classical ML algorithms, feature engineering. Deep learning is
deferred to a later stage.

Usage (run on your machine where the data lives):
    python run_ml.py --data-root . --protocol within --features bandpower
    python run_ml.py --data-root . --protocol within --features bandpower --subject-norm
    python run_ml.py --data-root . --protocol subjdep  --features bandpower   # pooled k-fold

Protocols:
    within  -> per-subject leave-one-situation-out; metrics averaged over
               subjects (TRUE within-subject; the recommended primary view).
    subjdep -> subject-dependent pooled stratified k-fold (same person may
               appear in train and test; the common DASPS "subject-dependent"
               setup, more training data, useful as a complementary view).
    loso    -> subject-independent (kept for comparison / honesty).

Features:
    bandpower -> per-channel mean log band-power (14 channels x 4 bands = 56)
                 -- compact, interpretable, standard for classical ML.
    tensor    -> full channel x band x time (14 x 4 x 9 = 504) -- richer, but
                 high-dimensional for tiny within-subject folds.
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor, epochs_to_connectivity
from validation import (loso_splits, mixed_splits, within_subject_splits,
                        fit_apply_standardizer, subject_normalize)

warnings.filterwarnings("ignore")
S = config.SEED


def make_models():
    """A range of traditional ML classifiers (supervisor's first stage)."""
    return {
        "LogReg":      LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
        "LDA":         LinearDiscriminantAnalysis(),
        "SVM-linear":  SVC(kernel="linear", class_weight="balanced", probability=False),
        "SVM-RBF":     SVC(kernel="rbf", class_weight="balanced", probability=False),
        "kNN":         KNeighborsClassifier(n_neighbors=5),
        "DecisionTree":DecisionTreeClassifier(class_weight="balanced", random_state=S),
        "RandomForest":RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=S),
        "ExtraTrees":  ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=S),
        "GradBoost":   GradientBoostingClassifier(random_state=S),
        "AdaBoost":    AdaBoostClassifier(random_state=S),
        "NaiveBayes":  GaussianNB(),
    }


def _scores(clf, X):
    """Probability-like score for the positive class (for AUC)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X)
        return (d - d.min()) / (np.ptp(d) + 1e-9)
    return clf.predict(X).astype(float)


def get_splits(protocol, subj, y):
    if protocol == "within":
        return list(within_subject_splits(subj, y))
    if protocol == "subjdep":
        return list(mixed_splits(subj, y, k=5))
    return list(loso_splits(subj))


def evaluate_model(clf_factory, flat, y, subj, splits, protocol):
    """Return dict of metrics. For 'within', metrics are averaged per subject."""
    oof_pred = np.full(len(y), -1)
    oof_score = np.full(len(y), np.nan)
    grp = np.full(len(y), -1)
    for tr, te, held in splits:
        Xtr, Xte = fit_apply_standardizer(flat[tr], flat[te])
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        oof_pred[te] = clf.predict(Xte)
        oof_score[te] = _scores(clf, Xte)
        grp[te] = held

    ev = oof_pred >= 0
    if protocol == "within":
        accs, f1s, aucs = [], [], []
        for s in np.unique(grp[ev]):
            m = ev & (grp == s)
            accs.append(accuracy_score(y[m], oof_pred[m]))
            f1s.append(f1_score(y[m], oof_pred[m], average="macro", zero_division=0))
            if len(np.unique(y[m])) == 2 and not np.isnan(oof_score[m]).any():
                aucs.append(roc_auc_score(y[m], oof_score[m]))
        return dict(acc=np.mean(accs), f1=np.mean(f1s),
                    auc=(np.mean(aucs) if aucs else np.nan),
                    auc_std=(np.std(aucs) if aucs else 0.0), n=len(accs))
    else:
        auc = (roc_auc_score(y[ev], oof_score[ev])
               if len(np.unique(y[ev])) == 2 and not np.isnan(oof_score[ev]).any()
               else np.nan)
        return dict(acc=accuracy_score(y[ev], oof_pred[ev]),
                    f1=f1_score(y[ev], oof_pred[ev], average="macro", zero_division=0),
                    auc=auc, auc_std=0.0, n=ev.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--protocol", default="within", choices=["within", "subjdep", "loso"])
    ap.add_argument("--features", default="bandpower",
                    choices=["bandpower", "tensor", "connectivity", "combined"])
    ap.add_argument("--task", default="binary", choices=["binary", "fourclass"])
    ap.add_argument("--subject-norm", action="store_true")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat subjdep k-fold with different seeds for mean+/-std")
    ap.add_argument("--permtest", action="store_true")
    args = ap.parse_args()

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task=args.task,
                                               label_source="SAM")
    else:
        print("No --data-root: synthetic demo.")
        EEG, y, subj, phase, meta = make_synthetic()

    print(f"Label={meta['label_source']}  epochs={len(y)}  subjects="
          f"{len(np.unique(subj))}  class balance={np.bincount(y).tolist()}")

    tensor, axis = epochs_to_tensor(EEG)
    bandpow = tensor.mean(axis=3).reshape(len(y), -1)          # [n, 14*4=56]
    if args.features == "bandpower":
        flat = bandpow
    elif args.features == "tensor":
        flat = tensor.reshape(len(y), -1)                      # [n, 504]
    elif args.features == "connectivity":
        conn, _ = epochs_to_connectivity(EEG)
        flat = conn.reshape(len(y), -1)                        # [n, 4*91=364]
    else:  # combined
        conn, _ = epochs_to_connectivity(EEG)
        flat = np.concatenate([bandpow, conn.reshape(len(y), -1)], axis=1)  # 56+364
    fdim = flat.shape[1]
    if args.subject_norm:
        flat = subject_normalize(flat, subj)
    print(f"Features={args.features} ({fdim} dims)  Protocol={args.protocol.upper()}"
          f"{'  +subject-norm' if args.subject_norm else ''}")

    models = make_models()
    do_repeat = (args.protocol == "subjdep" and args.repeats > 1)

    if do_repeat:
        from validation import mixed_splits
        acc_auc = {name: {"auc": [], "acc": [], "f1": []} for name in models}
        for r in range(args.repeats):
            sp = list(mixed_splits(subj, y, k=5, seed=S + r))
            for name in models:
                m = evaluate_model(lambda n=name: make_models()[n], flat, y, subj, sp, "subjdep")
                for k in ("auc", "acc", "f1"):
                    acc_auc[name][k].append(m[k])
        rows = [(name, dict(auc=np.mean(d["auc"]), auc_std=np.std(d["auc"]),
                            acc=np.mean(d["acc"]), f1=np.mean(d["f1"])))
                for name, d in acc_auc.items()]
        rows.sort(key=lambda r: -r[1]["auc"])
        print(f"  (averaged over {args.repeats} repeats of 5-fold)\n")
        print(f"  {'Model':<13}{'AUC(mean±std)':<18}{'Accuracy':<11}{'macro-F1':<10}")
        print("  " + "-" * 50)
        for name, m in rows:
            print(f"  {name:<13}{m['auc']:.3f}\u00b1{m['auc_std']:.2f}      {m['acc']:.3f}      {m['f1']:.3f}")
        best = rows[0][0]
    else:
        splits = get_splits(args.protocol, subj, y)
        if args.protocol == "within":
            used = len(np.unique([h for _, _, h in splits]))
            print(f"  within-subject: {used} subjects usable (2-class), "
                  f"{len(splits)} situation-folds total")
        print()
        rows = []
        for name in models:
            m = evaluate_model(lambda n=name: make_models()[n], flat, y, subj, splits, args.protocol)
            rows.append((name, m))
        rows.sort(key=lambda r: (-(r[1]["auc"] if not np.isnan(r[1]["auc"]) else -1)))
        head = "AUC" if args.protocol != "within" else "AUC(mean/subj)"
        print(f"  {'Model':<13}{head:<16}{'Accuracy':<11}{'macro-F1':<10}")
        print("  " + "-" * 48)
        for name, m in rows:
            auc = f"{m['auc']:.3f}" + (f"\u00b1{m['auc_std']:.2f}" if args.protocol == "within" else "")
            print(f"  {name:<13}{auc:<16}{m['acc']:.3f}      {m['f1']:.3f}")
        best = rows[0][0]
    if args.permtest:
        print(f"\n  Permutation check on best model ({best}) ...")
        rng = np.random.default_rng(S)
        yp = rng.permutation(y)
        sp = get_splits(args.protocol, subj, yp)
        pm = evaluate_model(lambda: make_models()[best], flat, yp, subj, sp, args.protocol)
        print(f"  permuted {best}: AUC={pm['auc']:.3f}  acc={pm['acc']:.3f} "
              f"(should sit near chance)")

    print(f"\n  Best: {best}. Compare its AUC to the permutation check above; "
          "clear separation = real signal.")


if __name__ == "__main__":
    main()
