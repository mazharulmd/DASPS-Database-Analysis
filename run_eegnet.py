"""
Deep-learning stage, fair test: EEGNet on the RAW signal with augmentation.

This is the strongest, most standard deep-learning attempt for EEG. Unlike the
band-power CNN, EEGNet learns its own temporal/spatial filters from the raw
signal, and on-the-fly augmentation (noise, time-shift, scaling) enlarges the
tiny training set. If EEGNet still does not beat the classical ~0.69-0.72 AUC,
the "deep learning does not help on this small dataset" conclusion is solid.

Explainability for a raw-signal model:
  * channel saliency  (grad x input -> which electrodes matter)
  * band occlusion    (remove each band -> AUC drop -> which bands matter)
Both saved as figures. (FAA validation remains on the working tree+SHAP model;
a chance-level model has no meaningful explanation.)

Usage (needs torch):
    python run_eegnet.py --data-root . --protocol subjdep --epochs 120 --figdir Figures
"""
from __future__ import annotations
import os, argparse, warnings
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import config
from data import load_dasps, make_synthetic
from features import _bandpass
from validation import loso_splits, mixed_splits
from config import BANDS, BAND_NAMES, DASPS_CHANNELS, SFREQ

warnings.filterwarnings("ignore")
S = config.SEED
N_CH = len(DASPS_CHANNELS)

import torch
import torch.nn as nn
from model import EEGNet
torch.manual_seed(S); np.random.seed(S)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def standardize_raw(train, test):
    """Per-channel z-score using train statistics. Inputs [N, C, T]."""
    mu = train.mean(axis=(0, 2), keepdims=True)
    sd = train.std(axis=(0, 2), keepdims=True) + 1e-7
    return (train - mu) / sd, (test - mu) / sd


def augment(xb):
    """On-the-fly augmentation for a batch tensor [N,1,C,T]."""
    out = xb.clone()
    if torch.rand(1).item() < 0.5:                       # time shift
        sh = int(torch.randint(-40, 41, (1,)).item())
        out = torch.roll(out, shifts=sh, dims=-1)
    if torch.rand(1).item() < 0.5:                       # gaussian noise
        out = out + 0.05 * torch.randn_like(out)
    if torch.rand(1).item() < 0.5:                       # amplitude scale
        out = out * (0.9 + 0.2 * torch.rand(1, device=out.device))
    return out


def train_eegnet(Xtr, ytr, n_ch, n_t, epochs, val_frac=0.15, batch=32):
    rng = np.random.default_rng(S)
    idx = rng.permutation(len(ytr)); nval = max(int(len(ytr) * val_frac), 4)
    vi, ti = idx[:nval], idx[nval:]
    Xt = torch.tensor(Xtr[ti], dtype=torch.float32, device=DEV).unsqueeze(1)
    yt = torch.tensor(ytr[ti], dtype=torch.long, device=DEV)
    Xv = torch.tensor(Xtr[vi], dtype=torch.float32, device=DEV).unsqueeze(1)
    yv = torch.tensor(ytr[vi], dtype=torch.long, device=DEV)

    cls, cnt = np.unique(ytr, return_counts=True)
    w = torch.tensor(cnt.sum() / (len(cls) * cnt), dtype=torch.float32, device=DEV)
    model = EEGNet(n_ch, n_t, n_classes=2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    best, bestv, pat, bad = None, 1e9, 20, 0
    n = len(ti)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for st in range(0, n, batch):
            b = perm[st:st + batch]
            opt.zero_grad()
            loss = lossf(model(augment(Xt[b])), yt[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), yv).item()
        if vl < bestv - 1e-4:
            bestv, best, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= pat:
                break
    if best:
        model.load_state_dict(best)
    model.eval(); return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--protocol", default="subjdep", choices=["subjdep", "loso"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--figdir", default="Figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task="binary", label_source="SAM")
    else:
        print("No --data-root: synthetic demo."); EEG, y, subj, phase, meta = make_synthetic()
    print(f"Label={meta['label_source']}  epochs={len(y)}  device={DEV}  "
          f"class balance={np.bincount(y).tolist()}")
    EEG = EEG.astype(np.float32)
    n_t = EEG.shape[2]

    # precompute band components for occlusion
    band_comp = {b: _bandpass(EEG, lo, hi).astype(np.float32)
                 for b, (lo, hi) in BANDS.items()}

    splits = list(mixed_splits(subj, y, k=5)) if args.protocol == "subjdep" else list(loso_splits(subj))
    print(f"Model=EEGNet(raw)  Protocol={args.protocol.upper()}  folds={len(splits)}  "
          f"epochs<= {args.epochs}  (augmentation on)\n")

    oof_pred = np.full(len(y), -1); oof_proba = np.full(len(y), np.nan)
    chan_sal = np.zeros(N_CH); band_drop = {b: [] for b in BAND_NAMES}; nf = 0
    for tr, te, held in splits:
        Xtr, Xte = standardize_raw(EEG[tr], EEG[te])
        model = train_eegnet(Xtr, y[tr], N_CH, n_t, args.epochs)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV).unsqueeze(1)
        with torch.no_grad():
            proba = torch.softmax(model(Xte_t), 1)[:, 1].cpu().numpy()
        oof_proba[te] = proba; oof_pred[te] = (proba >= 0.5).astype(int)

        # channel saliency (grad x input), averaged over test
        xs = Xte_t.clone().detach().requires_grad_(True)
        p = torch.softmax(model(xs), 1)[:, 1].sum(); model.zero_grad(); p.backward()
        sal = (xs.grad * xs).abs().mean(dim=(0, 1)).detach().cpu().numpy()  # [C,T]
        chan_sal += sal.sum(axis=1)

        # band occlusion: AUC drop when each band removed from test input
        base_auc = roc_auc_score(y[te], proba) if len(np.unique(y[te])) == 2 else np.nan
        mu = EEG[tr].mean(axis=(0, 2), keepdims=True); sd = EEG[tr].std(axis=(0, 2), keepdims=True) + 1e-7
        for b in BAND_NAMES:
            occ = ((EEG[te] - band_comp[b][te]) - mu) / sd
            ot = torch.tensor(occ, dtype=torch.float32, device=DEV).unsqueeze(1)
            with torch.no_grad():
                pp = torch.softmax(model(ot), 1)[:, 1].cpu().numpy()
            a = roc_auc_score(y[te], pp) if len(np.unique(y[te])) == 2 else np.nan
            band_drop[b].append(base_auc - a)
        nf += 1

    ev = oof_pred >= 0
    auc = roc_auc_score(y[ev], oof_proba[ev]) if len(np.unique(y[ev])) == 2 else float("nan")
    print("EEGNet performance (out-of-fold pooled):")
    print(f"  AUC={auc:.3f}  accuracy={accuracy_score(y[ev], oof_pred[ev]):.3f}  "
          f"macro-F1={f1_score(y[ev], oof_pred[ev], average='macro'):.3f}\n")

    chan_sal /= nf; chan_sal /= chan_sal.sum() + 1e-12
    print("Channel saliency (top 6):")
    for i in np.argsort(chan_sal)[::-1][:6]:
        print(f"  {DASPS_CHANNELS[i]:<5}{chan_sal[i]:.3f}")
    print("\nBand occlusion (AUC drop when band removed; higher = more important):")
    bd = {b: float(np.nanmean(v)) for b, v in band_drop.items()}
    for b in sorted(bd, key=lambda k: -bd[k]):
        print(f"  {b:<7}{bd[b]:+.3f}")

    if auc < 0.6:
        print("\n>>> EEGNet is at/near chance: its explanation is NOT meaningful and "
              "should not be interpreted. Conclusion: deep learning does not beat the "
              "classical model on this dataset. Report the SHAP/FAA result from the "
              "tree model as your explainability contribution.")
    else:
        print("\n>>> EEGNet learned signal: its channel/band explanation is worth "
              "reporting and comparing against the tree+SHAP result.")

    _figs(chan_sal, bd, args.figdir)


def _figs(chan_sal, bd, figdir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = np.argsort(chan_sal)[::-1]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.bar([DASPS_CHANNELS[i] for i in order], [chan_sal[i] for i in order], color="#2E75B6")
        ax.set_title("EEGNet channel saliency", color="#1F3864", fontweight="bold")
        ax.set_ylabel("importance"); plt.xticks(rotation=45, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        p1 = os.path.join(figdir, "eegnet_channel_saliency.png"); fig.savefig(p1, dpi=140); plt.close()

        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        bands = list(bd.keys())
        ax.bar(bands, [bd[b] for b in bands], color="#1F3864")
        ax.set_title("EEGNet band occlusion (AUC drop)", color="#1F3864", fontweight="bold")
        ax.set_ylabel("AUC drop when removed"); ax.axhline(0, color="#999", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        p2 = os.path.join(figdir, "eegnet_band_occlusion.png"); fig.savefig(p2, dpi=140); plt.close()
        print(f"\nFigures saved:\n  {p1}\n  {p2}")
    except Exception as e:
        print(f"\n(figures skipped: {e})")


if __name__ == "__main__":
    main()
