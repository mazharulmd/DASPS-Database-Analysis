# Explainable Anxiety Detection on DASPS — starter pipeline

A leak-free, subject-independent (LOSO) EEG anxiety-classification pipeline
whose explanations are reported over **electrode × frequency-band × time**
and **quantitatively validated against neurophysiology** (frontal alpha
asymmetry, frontal theta/beta). This is the foundation for the thesis; the
modelling is deliberately modest because the contribution is the
explanation-vs-neurophysiology validation, not raw accuracy.

## Why it's built this way

- **Three explanation axes are first-class.** Every epoch becomes a tensor
  `[channels(14) × bands(5) × time-windows]` of log band-power, so a single
  attribution pass yields importance per electrode, per band, and per time
  window — exactly what the topic promises.
- **No subject leakage.** Evaluation is Leave-One-Subject-Out; the held-out
  subject never appears in training, and all normalisation is fit on the
  training fold only. The numbers you get are the ones you can defend.
- **Explanations are tested, not just drawn.** `xai.py` runs statistical
  agreement tests: frontal concentration, band ranking, and FAA alignment.

## Traditional-ML stage (`run_ml.py`) — current supervisor-directed phase

SAM labels, within-subject analysis, preprocessed data, a panel of classical
classifiers (LogReg, LDA, linear/RBF SVM, kNN, Decision Tree, Random Forest,
Extra Trees, Gradient Boosting, AdaBoost, Naive Bayes), ranked by AUC.

```bash
python run_ml.py --data-root . --protocol within  --features bandpower --permtest
python run_ml.py --data-root . --protocol within  --features bandpower --subject-norm
python run_ml.py --data-root . --protocol subjdep --features bandpower --permtest
```

- `--protocol within` — per-subject leave-one-situation-out, metrics averaged
  over subjects (recommended primary view; truly within-subject).
- `--protocol subjdep` — subject-dependent pooled 5-fold (more training data;
  same person may appear in train and test — report as complementary).
- `--features bandpower` — 14×4 per-channel mean log band-power (compact,
  interpretable). `--features tensor` adds the time axis (504 dims).

Deep learning (`run_loso.py`, `BandTensorCNN`, SHAP/Grad-CAM) is deferred to a
later stage per supervisor guidance.

---

## Quick start

```bash
pip install -r requirements.txt
python run_loso.py --permtest            # synthetic smoke test (no data needed)
```

On synthetic data (which has a *planted* frontal alpha/beta signature) you
should see above-chance LOSO, permuted accuracy near 0.5, frontal
concentration PASS, beta among the top bands, and a positive FAA alignment —
i.e. the pipeline recovers known neurophysiology end-to-end.

## Running on real DASPS

The loader is wired to the exact Kaggle DASPS structure
(`DASPS_Database/Preprocessed data .mat/SNNpreprocessed.mat`, variable `data`
of shape `(12, 1920, 14)` = segments×time×channels, MATLAB v7.3; labels in
`participant_rating_public.xlsx`). Point `--data-root` at the **folder** or the
**.zip** directly — no manual unzip, no label CSV needed:

```bash
python run_loso.py --data-root /home/ubuntu/DASPS_Database --permtest
python run_loso.py --data-root /home/ubuntu/DASPS_Database --task fourclass
python run_loso.py --data-root /path/to/DASPS_Dataset.zip --hamilton h2
```

Labels (no guessing — taken straight from the file):
- **SAM per-situation** (`--label sam`) is the standard, more learnable DASPS
  target; thresholds follow the published scheme. `binary` = {normal,light}→0
  vs {moderate,severe}→1; situation = segment//2.
- **HAM-A severity per subject** (`--label hama`, default). `--hamilton h1`
  (better balance) or `h2`. With subject-level labels each LOSO fold is
  single-class, so read the **pooled** metrics for AUC.
- **`--subject-norm`**: per-subject (leak-free) feature z-scoring. In
  cross-subject EEG this is often the single biggest improvement — it removes
  the inter-subject offset that otherwise swamps the anxiety signal.

Recommended first sweep on real data:

```bash
python run_loso.py --data-root DASPS_Database --label hama --subject-norm --permtest
python run_loso.py --data-root DASPS_Database --label hama --hamilton h2 --subject-norm
python run_loso.py --data-root DASPS_Database --label sam  --subject-norm --permtest
python run_loso.py --data-root DASPS_Database --label sam
```

### Evaluation protocols (`--protocol`)
The leakage story is the thesis backbone. Run the same data three ways:

```bash
python run_loso.py --data-root . --label sam --protocol loso   --permtest  # honest
python run_loso.py --data-root . --label sam --protocol within --permtest  # per-person
python run_loso.py --data-root . --label sam --protocol mixed  --permtest  # LEAKY
python run_loso.py --data-root . --label hama --protocol mixed             # extreme leak
```

- **loso** — subject-independent, honest; your headline numbers.
- **within** — train/test on the same person (different segments); asks whether
  any anxiety *state* signal exists at all, even if it doesn't generalise.
- **mixed** — subject-agnostic k-fold; segments leak across train/test. This is
  what inflated DASPS papers effectively do. The gap mixed − loso *is* the
  leakage. With HAM-A (subject-level) labels, mixed approaches the trivial
  "recognise the person" ceiling.

Read the **pooled AUC** vs the **permuted** number. Honest near-chance is a
genuine finding, not a bug.

## Files

| File | Role |
|------|------|
| `config.py` | constants: montage, bands, FAA pairs, hyperparameters |
| `data.py` | real DASPS loader + synthetic generator (planted signal) |
| `features.py` | channel × band × time log-band-power tensor |
| `neuromarkers.py` | FAA, frontal theta/beta — computed independently of the model |
| `validation.py` | LOSO splitter + train-fold-only standardisation |
| `model.py` | `LinearBaseline` (sklearn) + `BandTensorCNN`/`EEGNet` (PyTorch) |
| `xai.py` | importance → channel/band/time profiles + neurophysiology tests |
| `run_loso.py` | end-to-end: LOSO training, honest metrics, validation report |

## Suggested path

1. **Run the synthetic test** and read every printed number until it makes
   sense — that is your leakage/credibility safety net.
2. **Plug in real DASPS**; reproduce the linear LOSO baseline first. This is
   your honest floor.
3. **Swap in `BandTensorCNN`** (needs `pip install torch`). Replace the
   linear model's `coef_importance()` with SHAP/Grad-CAM on the tensor; feed
   the resulting `[C,B,W]` map straight into `xai.run_validation`.
4. **Report agreements and disagreements** with FAA and frontal theta/beta —
   that comparison is the thesis contribution.

## Honesty notes for the write-up

- Report LOSO mean ± std and the permutation check, not a single best fold.
- Expect lower headline accuracy than leaky DASPS papers; say so explicitly.
- Treat the 4-class (severity) task cautiously — the severe class is tiny.
- Show that highlighted regions are *stable across folds* before claiming a
  brain region matters; single heatmaps on small noisy EEG are unreliable.
