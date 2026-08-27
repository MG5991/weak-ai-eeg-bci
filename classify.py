#!/usr/bin/env python3
"""
================================================================================
 classify.py  —  load a model.joblib bundle and classify WITHOUT retraining
================================================================================
Loads a bundle written by train_model.py, rebuilds features with the SAME
channel/band spec the bundle was trained with (via eeg_engine.build_features,
reused not reinvented), then applies the saved StandardScaler + estimator, and
writes a playback_*.json that EEG_control_prototype.html can animate.

SUBJECT-NORM CALIBRATION — read this before using a --subject-norm bundle:
A bundle trained with train_model.py --subject-norm was fit on features that
were z-scored PER PARTICIPANT using only that participant's own trials
(eeg_engine.per_subject_normalize). A raw feature vector is on the wrong scale
for that saved scaler/estimator unless the SAME per-subject z-scoring is redone
first. This script reproduces that step at classify-time: it groups the
incoming trials by participant, computes each participant's own mean/std from
ONLY their own trials in THIS run, and normalizes before handing off to the
saved scaler. This mirrors real deployment — a short per-subject calibration
recording, not a retrain — and matches exactly what eeg_engine.py / train_model.py
do (per_subject_normalize uses no labels, so it's valid on unseen subjects too).
Skipping this step for a --subject-norm bundle would silently produce garbage
predictions, since the estimator would see features on a completely different
scale than it was trained on.

Feature-size / montage guard: a model trained on one dataset/channel-montage
cannot classify a different-shaped one (e.g. an Emotiv-14ch bundle can't score
BCI-IV-2b's 3-channel montage) — this is checked explicitly and raises a clear
error rather than silently misclassifying.

Prints BOTH: accuracy on the data just classified, AND the bundle's stored
honest LOPO accuracy — the given-data number is resubstitution (optimistic) if
this is the bundle's own training data (train_model.py always fits on ALL
trials, so classifying that same dataset/config always is), so the honest
LOPO number is what should be reported/trusted, per this project's "honest
framing" convention (see CLAUDE.md).

Everything here is offline/virtual — no hardware, no MQTT.
================================================================================
"""
import argparse, json, os, re, sys
import numpy as np
import joblib

import eeg_engine as E


def main():
    ap = argparse.ArgumentParser(
        description="Classify a dataset with a persisted model.joblib bundle (no retraining).")
    ap.add_argument("--model", default="model.joblib", help="bundle written by train_model.py")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--dataset", choices=["2b", "physionet"])
    ap.add_argument("--playback", help="output playback json (default playback_classify_<dataset>.json)")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    meta, spec = bundle["metadata"], bundle["feature_spec"]
    print(f"Loaded bundle: model={meta['model_name']}  trained_on={meta['dataset']}  "
          f"channels={meta['channels']}  subject_norm={meta['subject_norm']}")

    # Channel selection MUST match what the bundle was trained with — use the
    # bundle's own spec, never a user-supplied value, so features line up.
    ns = argparse.Namespace(csv=a.csv, npz=None, raw=None, moabb=a.dataset,
                             channels=spec["channels_mode"], asymmetry=False)
    D = E.load_any(ns)
    print(f"Dataset to classify: {D['name']}  |  trials={len(D['y'])}  |  "
          f"participants={len(np.unique(D['groups']))}")

    Xfeat, feature_names = E.build_features(D, spec["channels_mode"])
    y = np.asarray(D["y"]); groups = np.asarray(D["groups"])

    # ---- guard: montage/feature-size mismatch -> clear error, no silent garbage ----
    if Xfeat.shape[1] != spec["n_features"]:
        sys.exit(
            f"Feature-size mismatch: {D['name']} [channels={spec['channels_mode']}] has "
            f"{Xfeat.shape[1]} features, but the bundle {a.model!r} was trained on "
            f"{spec['n_features']} features from {meta['dataset']} [channels={meta['channels']}]. "
            f"A model trained on one dataset/montage cannot classify a different-shaped one — "
            f"retrain with train_model.py on THIS dataset instead."
        )

    # ---- subject-norm calibration (see module docstring) ----
    if meta["subject_norm"]:
        Xfeat = E.per_subject_normalize(Xfeat, groups)
        print("Applied per-subject calibration (z-scored each participant's trials using only "
              "their own data in THIS run) to match the bundle's --subject-norm training.")

    Xs = bundle["scaler"].transform(Xfeat)
    clf = bundle["estimator"]
    pred = clf.predict(Xs)
    if hasattr(clf, "predict_proba"):
        conf = clf.predict_proba(Xs).max(1)
    else:
        conf = np.ones(len(pred))   # no probability estimate available for this estimator

    acc = float(np.mean(pred == y))
    print(f"\nAccuracy on this data: {acc:.3f}  ({len(y)} trials)")
    print("  NOTE (honest framing): if this is/contains the bundle's own training data, the "
          "number above is RESUBSTITUTION accuracy (optimistic — train_model.py always fits "
          "the deployment model on ALL trials, so classifying that same dataset/config is by "
          "definition resubstitution). The honest cross-subject estimate is the bundle's stored "
          f"LOPO accuracy: {meta['lopo_acc_mean']:.3f} (+/- {meta['lopo_acc_std']:.3f}, "
          f"{meta['lopo_protocol']}) — trust that number, not the one above.")

    classes_disp = [str(c) for c in np.unique(y)]
    if len(classes_disp) == 2:
        classes_disp = ["Down", "Up"]

    trials = [{"pred": int(pred[i]), "conf": round(float(conf[i]), 3),
               "true": int(y[i]), "participant": str(groups[i])} for i in range(len(y))]
    pb = {
        "dataset": D["name"],
        "protocol": f"classify.py replay of {meta['model_name']} from {os.path.basename(a.model)} (no retraining)",
        "n_trials": len(trials),
        "classes": classes_disp,
        "command_map": bundle["command_map"],
        "model": meta["model_name"],
        "accuracy": round(acc, 3),
        "trials": trials,
    }
    out = a.playback or f"playback_classify_{re.sub(r'\\W+', '_', D['name']).strip('_')}.json"
    with open(out, "w") as f:
        json.dump(pb, f)
    print(f"\nplayback written -> {out}   (accuracy {pb['accuracy']}, {pb['n_trials']} trials)")
    print("Load this file in EEG_control_prototype.html (button: 'Load playback') to animate it.")


if __name__ == "__main__":
    main()
