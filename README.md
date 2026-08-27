# Weak AI EEG-BCI

Software and cross-dataset benchmark for a PhD (candidate-of-science) research topic:
**"Models and Methods for Human–IoT interaction using Weak AI."**

"Weak AI" here means **narrow, task-specific** AI — not "low quality," and not "weak learner" in
the ensemble-methods sense. The system classifies motor-imagery (MI) EEG signals into binary
commands (left/right hand) that drive a **virtual** IoT device — a simulated drone or lamp.

The dissertation defense (aspirantura) for this work is already passed. This repository is the
post-defense benchmark and tooling that feeds a set of planned journal papers.

## What's here

- A feature-based EEG classification pipeline (band-power features, harmonized across datasets)
  evaluated under **Leave-One-Participant-Out (LOPO)** cross-validation — the honest way to measure
  cross-subject generalization for a BCI, as opposed to within-subject resubstitution accuracy.
- A model roster (Decision Tree, Random Forest, XGBoost/HistGradientBoosting, Shrinkage-LDA,
  Logistic Regression, MLP, SVM-RBF, Self-Training SVM-RBF) benchmarked across three datasets:
  an in-house Emotiv EPOC X recording, BCI Competition IV-2b, and PhysioNet Motor Imagery.
- Statistical significance testing (paired Wilcoxon signed-rank across folds, Holm-Bonferroni
  corrected) so that "best model" claims aren't just noise.
- Model persistence (`train_model.py` / `classify.py`) — train once, classify new data or replay a
  session without retraining.
- A browser-based control prototype (`EEG_control_prototype.html`) that animates a virtual
  drone/lamp from classifier output, with an IDLE fail-safe below a confidence threshold.
- A Streamlit app (`streamlit_app.py`) for exploring the benchmark results and running the
  classifier interactively.

## Honest framing (important)

Cross-subject accuracies on these datasets sit around **0.55–0.69** — above chance (0.50) but below
the ~0.70 threshold usually considered practical for real-time BCI control. Results here are
reported as **preliminary feasibility**, not a production-ready BCI system. One configuration
(BCI-IV-2b, all channels + subject-normalization) reaches 0.707, but this is a single 9-subject
dataset with low statistical power — it does not establish that the practical threshold is
generally cleared.

Self-training (semi-supervised learning with 20% labels) does **not** beat fully-supervised
training on any dataset tested here. It is presented as a **label-efficiency** result (you can get
close to supervised accuracy with far fewer labels), never as an accuracy win.

See `CLAUDE.md` and `summary_report.md` for the full methodology, caveats, and per-dataset results.

## Data

The in-house dataset (`eeg_dataset_emotiv.csv`) is a 14-channel Emotiv EPOC X recording, band-power
features only (no raw signals), with participant identifiers anonymized to `P01`–`P10`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Public datasets (BCI-IV-2b, PhysioNet MI) are downloaded on first use via
[MOABB](https://github.com/NeuroTechX/moabb) (needs `pip install moabb mne`, and one internet
connection to cache them).

## Usage

```bash
# in-house dataset, best-known config (subject-normalized features)
python eeg_engine.py --csv eeg_dataset_emotiv.csv --subject-norm

# public datasets
python eeg_engine.py --dataset 2b --channels all
python eeg_engine.py --dataset physionet --channels all

# refresh the cross-dataset summary report
python summarize.py

# train once, then classify/replay without retraining
python train_model.py --csv eeg_dataset_emotiv.csv --subject-norm --model lda --out model.joblib
python classify.py --model model.joblib --csv eeg_dataset_emotiv.csv

# regression test for model persistence
python test_persistence.py

# interactive dashboard + live classifier
streamlit run streamlit_app.py
```

Open `EEG_control_prototype.html` in a browser and use **Load playback** to animate the virtual
drone/lamp from any `playback_*.json` file produced by `eeg_engine.py` or `classify.py`.

## Repository layout

See `CLAUDE.md` for a full file-by-file breakdown, flags, and the complete set of verified results
across all datasets and configurations.
