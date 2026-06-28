"""
End-to-end runner: LOSO training + honest metrics + explanation validation.

Default run uses the synthetic generator (with planted frontal alpha/beta
signature) so you can confirm the whole pipeline works before touching real
data. Point --data-root at your DASPS download to run for real.

    python run_loso.py                 # synthetic smoke test
    python run_loso.py --data-root /path/to/DASPS --labels labels.csv

What it reports:
  * Mean +/- std accuracy / macro-F1 / AUC across the 23 LOSO folds (honest,
    subject-independent -- the numbers you put in your thesis).
  * A label-permutation sanity check (is the model genuinely above chance?).
  * The explanation-vs-neurophysiology report (frontal concentration, band
    ranking, FAA alignment), aggregated across folds.
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor, flatten_tensor
from validation import loso_splits, fit_apply_standardizer, subject_normalize
from model import LinearBaseline
from xai import flat_importance_to_map, run_validation


def _evaluate(flat, idx, y, subj, protocol="loso", permute=False, seed=config.SEED):
    """One full evaluation pass; returns (metrics, aggregated importance, idx).

    protocol: 'loso'   -> leave-one-subject-out (honest, subject-independent)
              'mixed'  -> subject-agnostic 5-fold (LEAKY; inflation demo)
              'within' -> per-subject leave-one-segment-out (within-person)
    """
    from validation import loso_splits, mixed_splits, within_subject_splits
    n_features = flat.shape[1]
    imp_accum = np.zeros(n_features)

    if permute:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    if protocol == "mixed":
        splits = list(mixed_splits(subj, y))
    elif protocol == "within":
        splits = list(within_subject_splits(subj, y))
    else:
        splits = list(loso_splits(subj))

    accs, f1s, aucs = [], [], []
    multiclass = len(np.unique(y)) > 2
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan)
    n_imp = 0
    for tr, te, held in splits:
        Xtr, Xte = fit_apply_standardizer(flat[tr], flat[te])
        clf = LinearBaseline(multiclass=multiclass).fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        pred = proba.argmax(1)
        oof_pred[te] = pred
        if not multiclass and proba.shape[1] == 2:
            oof_proba[te] = proba[:, 1]

        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        if not multiclass and len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], proba[:, 1]))
        imp_accum += clf.coef_importance()
        n_imp += 1

    imp_accum /= max(n_imp, 1)
    # pooled metrics over all evaluated segments
    ev = oof_pred >= 0
    pooled = dict(
        acc=accuracy_score(y[ev], oof_pred[ev]),
        f1=f1_score(y[ev], oof_pred[ev], average="macro", zero_division=0),
        auc=(roc_auc_score(y[ev], oof_proba[ev])
             if not multiclass and len(np.unique(y[ev])) == 2
             and not np.isnan(oof_proba[ev]).any() else float("nan")),
    )
    metrics = dict(
        acc=(np.mean(accs), np.std(accs)),
        f1=(np.mean(f1s), np.std(f1s)),
        auc=(np.mean(aucs), np.std(aucs)) if aucs else (float("nan"), 0.0),
        pooled=pooled,
        n_folds=len(accs),
    )
    return metrics, imp_accum, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="DASPS_Database folder OR the .zip; omit for synthetic")
    ap.add_argument("--task", default=config.TASK, choices=["binary", "fourclass"])
    ap.add_argument("--label", default="hama", choices=["hama", "sam"],
                    help="label source: HAM-A (per-subject) or SAM (per-situation)")
    ap.add_argument("--hamilton", default="h1", choices=["h1", "h2"],
                    help="which Hamilton measure to label from (HAM-A only)")
    ap.add_argument("--subject-norm", action="store_true",
                    help="per-subject feature z-scoring (leak-free calibration)")
    ap.add_argument("--protocol", default="loso",
                    choices=["loso", "mixed", "within"],
                    help="loso=honest; mixed=leaky inflation demo; within=per-subject")
    ap.add_argument("--permtest", action="store_true",
                    help="also run a label-permutation chance check")
    args = ap.parse_args()

    if args.data_root:
        print(f"Loading real DASPS from {args.data_root} ...")
        EEG, y, subj, phase, meta = load_dasps(
            args.data_root, task=args.task,
            label_source=args.label.upper(), hamilton=args.hamilton)
    else:
        print("No --data-root given: running SYNTHETIC smoke test "
              "(planted frontal alpha asymmetry + frontal beta).")
        EEG, y, subj, phase, meta = make_synthetic()

    print(f"  epochs={len(y)}  subjects={len(np.unique(subj))}  "
          f"channels={EEG.shape[1]}  label={meta['label_source']}  "
          f"class balance={np.bincount(y).tolist()}")

    print("Building channel x band x time tensor ...")
    tensor, axis_info = epochs_to_tensor(EEG)
    print(f"  tensor shape (per epoch): {axis_info['shape']} -> "
          f"{tensor.shape[1:]}  ({len(axis_info['windows'])} windows, "
          f"{len(axis_info['bands'])} bands)")

    flat, idx = flatten_tensor(tensor)
    if args.subject_norm:
        flat = subject_normalize(flat, subj)
        print("  applied per-subject normalisation (leak-free)")

    print(f"\nRunning protocol = {args.protocol.upper()} ...")
    metrics, imp, idx = _evaluate(flat, idx, y, subj, protocol=args.protocol)
    print(f"  accuracy : {metrics['acc'][0]:.3f} +/- {metrics['acc'][1]:.3f}")
    print(f"  macro-F1 : {metrics['f1'][0]:.3f} +/- {metrics['f1'][1]:.3f}")
    if not np.isnan(metrics["auc"][0]):
        print(f"  AUC      : {metrics['auc'][0]:.3f} +/- {metrics['auc'][1]:.3f}")
    po = metrics["pooled"]
    print(f"  pooled (all evaluated segments): acc={po['acc']:.3f}  "
          f"macroF1={po['f1']:.3f}  AUC={po['auc']:.3f}")
    print(f"  folds         : {metrics['n_folds']}")

    if args.permtest:
        print("\nLabel-permutation chance check ...")
        pm, _, _ = _evaluate(flat, idx, y, subj, protocol=args.protocol,
                             permute=True)
        print(f"  permuted pooled acc : {pm['pooled']['acc']:.3f}  "
              f"AUC={pm['pooled']['auc']:.3f} (should sit near chance)")

    # ----- explanation validation (importance aggregated across folds) -----
    imp_map = flat_importance_to_map(imp, idx, tensor.shape[1:])
    run_validation(imp_map, EEG, y, axis_info, verbose=True)

    print("\nNote: honest subject-independent LOSO on 23 subjects is hard; "
          "near-chance with the linear floor is a real finding, not a bug. "
          "Compare label sources (--label sam) and --subject-norm, then add "
          "the deep model.")


if __name__ == "__main__":
    main()
