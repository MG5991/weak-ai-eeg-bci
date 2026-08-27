#!/usr/bin/env python3
"""
================================================================================
 train_model.py  —  persist ONE eeg_engine.py roster model as model.joblib
================================================================================
Trains a single model (your choice of --model) on ALL trials of a dataset, for
deployment, while separately computing the honest cross-subject accuracy using
eeg_engine.lopo_single_model() — the SAME per-fold Leave-One-Participant-Out
protocol used by eeg_engine.py's comparison table — so the number stored in the
bundle is directly comparable to results_master.csv / summary_report.md. It is
NOT the resubstitution accuracy of the deployment fit (that would be optimistic).

Saved bundle (joblib): fitted estimator, fitted StandardScaler, feature spec
(channel mode, bands, ordered feature names), class->command map, and metadata
(model, dataset, channels, subject-norm flag, honest LOPO acc+std, n_train,
sklearn version, timestamp).

SUBJECT-NORM GOTCHA: if --subject-norm is used, the persisted scaler/estimator
are fit on features that were z-scored PER PARTICIPANT using only that
participant's own trials (eeg_engine.per_subject_normalize). A brand-new
subject's raw features are on the wrong scale for this bundle — classify.py
must redo that same per-subject z-scoring from the new subject's own trials
before it can use the saved scaler/estimator. See classify.py's docstring.

Everything here is offline/virtual — no hardware, no MQTT.
================================================================================
"""
import argparse, datetime
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import eeg_engine as E
import mi_experiment as M


def main():
    ap = argparse.ArgumentParser(
        description="Train and persist one eeg_engine.py roster model as a deployable model.joblib bundle.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="feature CSV (e.g. the Emotiv dataset; cols ending __feat + movement + participant)")
    src.add_argument("--dataset", choices=["2b", "physionet"], help="public MOABB dataset (raw epochs)")
    ap.add_argument("--channels", choices=["all", "motor", "motor-wide"], default="all",
                     help="channel subset, same semantics as eeg_engine.py --channels")
    ap.add_argument("--subject-norm", dest="subject_norm", action="store_true",
                     help="z-score each participant's features using only their own trials before "
                          "pooling/training — matches eeg_engine.py --subject-norm. The saved bundle "
                          "records this flag; classify.py MUST redo per-subject calibration at "
                          "inference time when it is set (see classify.py docstring).")
    ap.add_argument("--model", required=True, choices=["svm", "lda", "logreg", "rf", "xgb", "mlp", "dt"],
                     help="which roster model to train and persist")
    ap.add_argument("--out", default="model.joblib", help="output bundle path (default model.joblib)")
    a = ap.parse_args()

    # Reuse eeg_engine's own loader — same code path eeg_engine.py itself uses.
    ns = argparse.Namespace(csv=a.csv, npz=None, raw=None, moabb=a.dataset,
                             channels=a.channels, asymmetry=False)
    D = E.load_any(ns)
    print(f"\nDataset: {D['name']}  |  trials={len(D['y'])}  |  participants={len(np.unique(D['groups']))}")

    Xfeat, feature_names = E.build_features(D, a.channels)
    y = np.asarray(D["y"]); groups = np.asarray(D["groups"])

    if a.subject_norm:
        Xfeat = E.per_subject_normalize(Xfeat, groups)
        print("Applied per-subject normalization (z-score each participant using their own trials).")

    # ---- honest cross-subject accuracy: reuse the engine's LOPO logic verbatim ----
    full_name, acc_mean, acc_std, proto = E.lopo_single_model(Xfeat, y, groups, a.model)
    print(f"Honest cross-subject accuracy [{proto}]: {full_name} = {acc_mean:.3f} (+/- {acc_std:.3f})")

    # ---- deployment fit: SAME model, but trained on ALL trials (no held-out subject) ----
    scaler = StandardScaler().fit(Xfeat)
    Xs = scaler.transform(Xfeat)
    if a.model == "svm":
        # probability=True here (unlike the roster's SVM, kept probability=False for
        # speed in the full comparison table) so a persisted SVM bundle can report a
        # prediction confidence in classify.py. predict() itself is identical either
        # way, so this does not change the honest LOPO number computed above.
        deploy_clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    else:
        _, builder = E.get_model_builders()[a.model]
        deploy_clf = builder()
    clf = deploy_clf.fit(Xs, y)

    classes = np.unique(y)
    class_names = D.get("classnames")
    if class_names and len(class_names) == len(classes):
        command_map = {str(i): class_names[i] for i in range(len(classes))}
    elif len(classes) == 2:
        command_map = {"0": "DOWN", "1": "UP"}
    else:
        command_map = {str(int(c)): str(int(c)) for c in classes}

    bundle = {
        "estimator": clf,
        "scaler": scaler,
        "feature_spec": {
            "channels_mode": a.channels,
            "bands": [b for b, _, _ in M.BANDS],
            "feature_names": feature_names,
            "n_features": int(Xfeat.shape[1]),
        },
        "command_map": command_map,
        "metadata": {
            "model_key": a.model,
            "model_name": full_name,
            "dataset": D["name"],
            "channels": a.channels,
            "subject_norm": bool(a.subject_norm),
            "lopo_protocol": proto,
            "lopo_acc_mean": round(acc_mean, 4),
            "lopo_acc_std": round(acc_std, 4),
            "n_train": int(len(y)),
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    joblib.dump(bundle, a.out)
    print(f"\nSaved model bundle -> {a.out}")
    print(f"  model={full_name}  dataset={D['name']}  channels={a.channels}  subject_norm={a.subject_norm}")
    print(f"  honest LOPO accuracy = {acc_mean:.3f} (+/- {acc_std:.3f}, {proto})  [fit on n_train={len(y)} trials]")


if __name__ == "__main__":
    main()
