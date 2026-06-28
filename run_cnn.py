"""
Deep-learning stage: a compact CNN on the channel x band x time tensor, with
explainability (Grad-CAM + gradient input-attribution) validated against
frontal alpha asymmetry (FAA). This is the thesis core.

Honest expectations: with only 276 epochs from 23 subjects, a CNN overfits
easily and is unlikely to beat the ~0.69-0.72 AUC of the classical models.
The value here is the EXPLANATION from a deep model, not a higher score.

Usage (needs PyTorch: pip install torch):
    python run_cnn.py --data-root . --protocol subjdep --epochs 60 --figdir Figures

Outputs:
  * pooled AUC / accuracy / macro-F1 (out-of-fold)
  * explanation-vs-neurophysiology report (frontal / bands / FAA)
  * Figures/: Grad-CAM (electrode x time) and attribution (electrode x band)
"""
from __future__ import annotations
import os, argparse, warnings
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import config
from data import load_dasps, make_synthetic
from features import epochs_to_tensor
from validation import (loso_splits, mixed_splits, within_subject_splits,
                        fit_apply_standardizer)
from xai import run_validation
from config import BAND_NAMES, DASPS_CHANNELS

warnings.filterwarnings("ignore")
S = config.SEED
N_CH, N_BAND = len(DASPS_CHANNELS), len(BAND_NAMES)

import torch
import torch.nn as nn
from model import BandTensorCNN
torch.manual_seed(S)
np.random.seed(S)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def get_splits(protocol, subj, y):
    if protocol == "within":
        return list(within_subject_splits(subj, y))
    if protocol == "loso":
        return list(loso_splits(subj))
    return list(mixed_splits(subj, y, k=5))


def train_fold(Xtr, ytr, n_bands, n_ch, n_win, epochs, val_frac=0.15, batch_size=32):
    """Train BandTensorCNN with class weights, MINI-BATCH SGD, early stopping."""
    rng = np.random.default_rng(S)
    idx = rng.permutation(len(ytr))
    nval = max(int(len(ytr) * val_frac), 4)
    vi, ti = idx[:nval], idx[nval:]
    Xt = torch.tensor(Xtr[ti], dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr[ti], dtype=torch.long, device=DEV)
    Xv = torch.tensor(Xtr[vi], dtype=torch.float32, device=DEV)
    yv = torch.tensor(ytr[vi], dtype=torch.long, device=DEV)

    cls, cnt = np.unique(ytr, return_counts=True)
    w = torch.tensor((cnt.sum() / (len(cls) * cnt)), dtype=torch.float32, device=DEV)
    model = BandTensorCNN(n_bands, n_ch, n_win, n_classes=2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    best_state, best_val, patience, bad = None, 1e9, 25, 0
    n = len(ti)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for start in range(0, n, batch_size):          # mini-batches => many updates
            b = perm[start:start + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xv), yv).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def attributions(model, Xte_t):
    """Gradient x input attribution per element -> mean over samples [B,C,W]."""
    x = Xte_t.clone().detach().requires_grad_(True)
    prob = torch.softmax(model(x), dim=1)[:, 1].sum()
    model.zero_grad()
    prob.backward()
    attr = (x.grad * x).abs().mean(dim=0).detach().cpu().numpy()  # [B,C,W]
    return attr


def grad_cam(model, Xte_t):
    """Grad-CAM on the last conv block -> [C, W] map averaged over samples."""
    acts, grads = {}, {}
    layer = model.features[9]  # ELU after last conv (64 feature maps)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0].detach()))
    x = Xte_t.clone().detach()
    out = model(x)
    score = out[:, 1].sum()
    model.zero_grad(); score.backward()
    A, G = acts["a"], grads["g"]                       # [N,64,C,W]
    alpha = G.mean(dim=(2, 3), keepdim=True)           # [N,64,1,1]
    cam = torch.relu((alpha * A).sum(dim=1))           # [N,C,W]
    h1.remove(); h2.remove()
    return cam.mean(0).cpu().numpy()                   # [C,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--protocol", default="subjdep", choices=["subjdep", "loso", "within"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--figdir", default="Figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    if args.data_root:
        EEG, y, subj, phase, meta = load_dasps(args.data_root, task="binary", label_source="SAM")
    else:
        print("No --data-root: synthetic demo."); EEG, y, subj, phase, meta = make_synthetic()
    print(f"Label={meta['label_source']}  epochs={len(y)}  device={DEV}  "
          f"class balance={np.bincount(y).tolist()}")

    tensor, _ = epochs_to_tensor(EEG)                  # [N,14,4,9]
    X = np.transpose(tensor, (0, 2, 1, 3))             # -> [N, bands(4), ch(14), win(9)]
    Bn, Cn, Wn = X.shape[1], X.shape[2], X.shape[3]
    flatshape = X.reshape(len(y), -1).shape[1]

    splits = get_splits(args.protocol, subj, y)
    print(f"Model=BandTensorCNN  Protocol={args.protocol.upper()}  "
          f"folds={len(splits)}  epochs<= {args.epochs}\n")

    oof_pred = np.full(len(y), -1); oof_proba = np.full(len(y), np.nan)
    attr_sum = np.zeros((Bn, Cn, Wn)); cam_sum = np.zeros((Cn, Wn)); nfold = 0
    for tr, te, held in splits:
        # standardise on train stats only (flatten -> scale -> reshape)
        ftr = X[tr].reshape(len(tr), -1); fte = X[te].reshape(len(te), -1)
        ftr, fte = fit_apply_standardizer(ftr, fte)
        Xtr = ftr.reshape(len(tr), Bn, Cn, Wn); Xte = fte.reshape(len(te), Bn, Cn, Wn)

        model = train_fold(Xtr, y[tr], Bn, Cn, Wn, args.epochs)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            proba = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()
        oof_proba[te] = proba; oof_pred[te] = (proba >= 0.5).astype(int)

        attr_sum += attributions(model, Xte_t)
        cam_sum += grad_cam(model, Xte_t)
        nfold += 1

    ev = oof_pred >= 0
    auc = roc_auc_score(y[ev], oof_proba[ev]) if len(np.unique(y[ev])) == 2 else float("nan")
    print("CNN performance (out-of-fold pooled):")
    print(f"  AUC={auc:.3f}  accuracy={accuracy_score(y[ev], oof_pred[ev]):.3f}  "
          f"macro-F1={f1_score(y[ev], oof_pred[ev], average='macro'):.3f}\n")

    # ---- explanation: attribution [B,C,W] -> [C,B,W] for the FAA validator ----
    attr = attr_sum / nfold                            # [B,C,W]
    imp_map = np.transpose(attr, (1, 0, 2))            # [C,B,W]
    axis_info = dict(channels=list(DASPS_CHANNELS), bands=list(BAND_NAMES),
                     windows=[(round(1.5 * k, 1), round(1.5 * k + 3, 1)) for k in range(Wn)],
                     shape="[channels, bands, windows]")
    run_validation(imp_map, EEG, y, axis_info, verbose=True)

    _save_figs(imp_map, cam_sum / nfold, args.figdir)


def _save_figs(imp_map, cam, figdir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # electrode x band attribution
        M = imp_map.sum(axis=2)                         # [C,B]
        fig, ax = plt.subplots(figsize=(4, 7))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(N_BAND)); ax.set_xticklabels(BAND_NAMES)
        ax.set_yticks(range(N_CH)); ax.set_yticklabels(DASPS_CHANNELS)
        ax.set_title("CNN attribution\n(electrode x band)", color="#1F3864", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.6); fig.tight_layout()
        p1 = os.path.join(figdir, "cnn_attribution_elec_band.png"); fig.savefig(p1, dpi=130); plt.close()
        # Grad-CAM electrode x time
        fig, ax = plt.subplots(figsize=(5, 6))
        im = ax.imshow(cam, aspect="auto", cmap="magma")
        ax.set_xticks(range(cam.shape[1])); ax.set_xticklabels([f"{k+1}" for k in range(cam.shape[1])])
        ax.set_yticks(range(N_CH)); ax.set_yticklabels(DASPS_CHANNELS)
        ax.set_xlabel("time window"); ax.set_title("Grad-CAM\n(electrode x time)", color="#1F3864", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.6); fig.tight_layout()
        p2 = os.path.join(figdir, "cnn_gradcam_elec_time.png"); fig.savefig(p2, dpi=130); plt.close()
        print(f"\nFigures saved:\n  {p1}\n  {p2}")
    except Exception as e:
        print(f"\n(figures skipped: {e})")


if __name__ == "__main__":
    main()
