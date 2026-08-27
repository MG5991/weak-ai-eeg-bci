#!/usr/bin/env python3
"""
================================================================================
 streamlit_app.py — interactive dashboard + live classifier for the EEG-MI
 cross-dataset benchmark ("Models and Methods for Human-IoT interaction using
 Weak AI"). Everything here is offline/virtual — no hardware, no MQTT.
================================================================================
Tab 1 (Results): renders results_master.csv / summary_report.md as charts and
tables, with the same */~ significance convention used everywhere else in this
project (paired Wilcoxon across folds, Holm-Bonferroni corrected).

Tab 2 (Live classifier): upload a feature CSV (or use the shipped Emotiv
bundle) and run it through a persisted model.joblib bundle via classify.py's
own logic (imported directly, not subprocessed) — including the
--subject-norm per-subject calibration step when the bundle needs it. Always
shows the bundle's honest LOPO accuracy next to the given-data accuracy,
since the latter is resubstitution whenever the data is the bundle's own
training set (see CLAUDE.md "Honest framing is mandatory").
"""
import argparse
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import eeg_engine as E
from significance import pairwise_significance

st.set_page_config(page_title="Weak AI EEG-BCI", page_icon="🧠", layout="wide")

RESULTS_CSV = "results_master.csv"
FOLDS_JSON = "folds_master.json"
SUMMARY_MD = "summary_report.md"
DEFAULT_BUNDLE = "model_emotiv_lda_subjnorm.joblib"

st.title("🧠 Weak AI EEG-BCI — motor-imagery classification benchmark")
st.caption(
    "PhD topic: *Models and Methods for Human–IoT interaction using Weak AI*. "
    "Everything below is offline/virtual — the classifier output drives a "
    "**simulated** drone/lamp, never real hardware."
)

tab_results, tab_live = st.tabs(["📊 Benchmark results", "🎛️ Live classifier"])

# ============================================================================
# Tab 1 — Results dashboard
# ============================================================================
with tab_results:
    if not os.path.exists(RESULTS_CSV):
        st.error(f"{RESULTS_CSV} not found — run `eeg_engine.py` at least once first.")
    else:
        df = pd.read_csv(RESULTS_CSV)

        st.warning(
            "**Honest framing.** Cross-subject accuracies here sit around 0.55-0.69 — "
            "above chance (0.50) but below the ~0.70 threshold usually considered "
            "practical for real-time BCI control. Treat results as **preliminary "
            "feasibility**, not a production-ready system. Self-training (20% labels) "
            "never beats fully-supervised training on these datasets — it is a "
            "label-efficiency result, not an accuracy win.",
            icon="⚠️",
        )

        configs = df[["dataset", "channels"]].drop_duplicates().reset_index(drop=True)
        configs["label"] = configs["dataset"] + "  [" + configs["channels"] + "]"
        choice = st.selectbox("Dataset / channel configuration", configs["label"])
        sel = configs[configs["label"] == choice].iloc[0]
        sub = df[(df["dataset"] == sel["dataset"]) & (df["channels"] == sel["channels"])].copy()
        sub = sub.sort_values("acc_mean", ascending=False)

        sig = {}
        if os.path.exists(FOLDS_JSON):
            import json
            folds = json.load(open(FOLDS_JSON))
            key = f"{sel['dataset']}|{sel['channels']}"
            if key in folds:
                model_accs = {m: d["acc"] for m, d in folds[key]["models"].items()}
                if len(model_accs) > 1:
                    sig = pairwise_significance(model_accs)

        def mark(row):
            m = row["model"]
            if m in sig:
                return "best" if sig[m]["is_best"] else ("tied" if sig[m]["tied_with_best"] else "")
            return ""

        sub["significance"] = sub.apply(mark, axis=1)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"{sel['dataset']} — {sel['channels']}")
            st.bar_chart(sub.set_index("model")["acc_mean"])
        with c2:
            st.metric("Participants", int(sub["participants"].iloc[0]))
            st.metric("Trials", int(sub["trials"].iloc[0]))
            st.metric("Best model", sub.iloc[0]["model"], f"{sub.iloc[0]['acc_mean']:.3f}")

        st.dataframe(
            sub[["model", "acc_mean", "acc_std", "macro_f1", "significance"]]
            .rename(columns={"acc_mean": "acc (mean)", "acc_std": "acc (std)", "macro_f1": "macro-F1"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "`best` = highest mean accuracy in this config · `tied` = not significantly "
            "different from best (paired Wilcoxon across folds, Holm-Bonferroni corrected, "
            "alpha=0.05) — blank means significantly worse than best."
        )

        st.subheader("Full cross-dataset report")
        if os.path.exists(SUMMARY_MD):
            with open(SUMMARY_MD) as f:
                st.markdown(f.read())
        else:
            st.info("Run `python summarize.py` to generate summary_report.md.")

# ============================================================================
# Tab 2 — Live classifier
# ============================================================================
with tab_live:
    st.subheader("Classify EEG feature data with a persisted model bundle")
    st.caption(
        "Reuses `classify.py`'s own logic (no retraining). If the bundle was trained "
        "with `--subject-norm`, per-subject z-score calibration is redone here on the "
        "uploaded data's own trials before scoring — a subject can't be scored on raw "
        "features against a subject-norm-trained scaler."
    )

    bundle_source = st.radio(
        "Model bundle", ["Shipped Emotiv bundle (Shrinkage-LDA, subject-norm)", "Upload a model.joblib"],
        horizontal=True,
    )
    bundle = None
    bundle_path = None
    if bundle_source.startswith("Shipped"):
        if os.path.exists(DEFAULT_BUNDLE):
            bundle_path = DEFAULT_BUNDLE
            bundle = joblib.load(bundle_path)
        else:
            st.error(f"{DEFAULT_BUNDLE} not found in the repo.")
    else:
        up_bundle = st.file_uploader("model.joblib", type=["joblib"])
        if up_bundle is not None:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
                tf.write(up_bundle.getvalue())
                bundle_path = tf.name
            bundle = joblib.load(bundle_path)

    if bundle is not None:
        meta, spec = bundle["metadata"], bundle["feature_spec"]
        b1, b2, b3 = st.columns(3)
        b1.metric("Model", meta["model_name"])
        b2.metric("Trained on", meta["dataset"])
        b3.metric("Honest LOPO accuracy", f"{meta['lopo_acc_mean']:.3f} ± {meta['lopo_acc_std']:.3f}")
        st.caption(
            f"channels={meta['channels']} · subject_norm={meta['subject_norm']} · "
            f"n_train={meta['n_train']} · {meta['lopo_protocol']}"
        )

        st.markdown("---")
        up_csv = st.file_uploader(
            "Feature CSV to classify (columns ending `__feat`, plus `movement` and `participant`)",
            type=["csv"],
        )
        if up_csv is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                tf.write(up_csv.getvalue())
                csv_path = tf.name

            ns = argparse.Namespace(csv=csv_path, npz=None, raw=None, moabb=None,
                                     channels=spec["channels_mode"], asymmetry=False)
            try:
                D = E.load_any(ns)
                Xfeat, feature_names = E.build_features(D, spec["channels_mode"])
                y = np.asarray(D["y"]); groups = np.asarray(D["groups"])

                if Xfeat.shape[1] != spec["n_features"]:
                    st.error(
                        f"Feature-size mismatch: this data has {Xfeat.shape[1]} features, but the "
                        f"bundle was trained on {spec['n_features']} features from "
                        f"{meta['dataset']} [channels={meta['channels']}]. A model trained on one "
                        f"dataset/montage cannot classify a different-shaped one."
                    )
                else:
                    if meta["subject_norm"]:
                        Xfeat = E.per_subject_normalize(Xfeat, groups)
                        st.info("Applied per-subject calibration (z-scored each participant's "
                                "trials using only their own data in this upload).")

                    Xs = bundle["scaler"].transform(Xfeat)
                    clf = bundle["estimator"]
                    pred = clf.predict(Xs)
                    conf = clf.predict_proba(Xs).max(1) if hasattr(clf, "predict_proba") else np.ones(len(pred))
                    acc = float(np.mean(pred == y))

                    r1, r2 = st.columns(2)
                    r1.metric("Accuracy on this data", f"{acc:.3f}", help="Resubstitution if this "
                              "is the bundle's own training data — not a generalization estimate.")
                    r2.metric("Trust instead: honest LOPO accuracy", f"{meta['lopo_acc_mean']:.3f}")

                    command_map = bundle["command_map"]
                    out = pd.DataFrame({
                        "participant": groups,
                        "true": y,
                        "predicted": pred,
                        "command": [command_map.get(str(int(p)), str(int(p))) for p in pred],
                        "confidence": np.round(conf, 3),
                        "correct": pred == y,
                    })
                    st.dataframe(out, use_container_width=True, hide_index=True)
            except SystemExit as e:
                st.error(str(e))
            finally:
                os.unlink(csv_path)
        else:
            st.caption("Upload a CSV above to classify it. Try the shipped "
                       "`eeg_dataset_emotiv.csv` for a quick demo (participant IDs are "
                       "anonymized, P01-P10).")
