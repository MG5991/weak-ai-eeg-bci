#!/usr/bin/env python3
"""
================================================================================
 test_persistence.py — regression test: does --subject-norm calibration
 actually generalize to an unseen participant?
================================================================================
Exercises train_model.py and classify.py exactly as a user would (subprocess,
real CLI), simulating the real deployment scenario: train a --subject-norm
bundle on N-1 participants, then classify the ONE participant who was NEVER
in training. The per-subject calibration in classify.py must z-score that new
participant using only their own trials before scoring — if that step were
broken (e.g. accidentally reused a training subject's stats, or skipped
normalization entirely), held-out accuracy would either collapse toward
chance or, more insidiously, silently look fine while actually being wrong.

What "correct" looks like, per this project's honest-framing convention
(see CLAUDE.md): the MEAN accuracy across held-out participants should track
the honest cross-subject LOPO estimate that train_model.py itself reports
(~0.65-0.66 for Shrinkage-LDA + --subject-norm on the Emotiv CSV) — and stay
well below the RESUBSTITUTION accuracy you'd get classifying a bundle's own
training data (~0.815 for this same model/config — see prior manual check).
Landing near resubstitution instead of LOPO would mean the calibration is
leaking information and inflating accuracy; landing far below LOPO would mean
it's broken and destroying signal.

Run:
  python test_persistence.py            # full leave-one-participant-out (10 participants, ~1 min)
  python test_persistence.py --quick    # 3 participants, looser tolerance (fast iteration)

Exit code 0 = pass, 1 = fail (with a printed reason).
================================================================================
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd
import joblib

CSV = "eeg_dataset_emotiv.csv"
MODEL = "lda"   # Shrinkage-LDA — the documented best model for Emotiv + --subject-norm
# Resubstitution accuracy for this exact model/config, classifying a bundle's
# own training data (see manual check in this session) — the honest, held-out
# mean must stay clearly below this or calibration is leaking.
RESUB_CEILING = 0.75


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r


def held_out_accuracy(df, participant, tmpdir):
    """Train on everyone EXCEPT participant, classify participant alone, return accuracy."""
    train_csv = os.path.join(tmpdir, f"train_{participant}.csv")
    test_csv = os.path.join(tmpdir, f"test_{participant}.csv")
    df[df["participant"] != participant].to_csv(train_csv, index=False)
    df[df["participant"] == participant].to_csv(test_csv, index=False)
    bundle = os.path.join(tmpdir, f"model_{participant}.joblib")
    playback = os.path.join(tmpdir, f"playback_{participant}.json")

    run([sys.executable, "train_model.py", "--csv", train_csv, "--subject-norm",
         "--model", MODEL, "--out", bundle])
    run([sys.executable, "classify.py", "--model", bundle, "--csv", test_csv,
         "--playback", playback])
    return json.load(open(playback))["accuracy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                     help="test only 3 participants with a looser tolerance (fast iteration)")
    a = ap.parse_args()

    df = pd.read_csv(CSV)
    participants = sorted(df["participant"].unique())
    tolerance = 0.10
    if a.quick:
        participants = participants[:3]
        tolerance = 0.20

    with tempfile.TemporaryDirectory() as tmpdir:
        # Reference: the honest LOPO accuracy train_model.py itself reports when
        # trained on ALL participants — what we expect held-out accuracy to track.
        full_bundle = os.path.join(tmpdir, "model_full.joblib")
        run([sys.executable, "train_model.py", "--csv", CSV, "--subject-norm",
             "--model", MODEL, "--out", full_bundle])
        reference_lopo = joblib.load(full_bundle)["metadata"]["lopo_acc_mean"]
        print(f"Reference honest LOPO ({MODEL} + --subject-norm, all {len(df['participant'].unique())} "
              f"participants, from train_model.py): {reference_lopo:.3f}\n")

        print(f"Held-out-participant deployment check ({len(participants)} participant(s), "
              f"train on the rest, classify the one left out):")
        results = {}
        for p in participants:
            acc = held_out_accuracy(df, p, tmpdir)
            results[p] = acc
            print(f"  held out {p:<12} -> accuracy {acc:.3f}")

    mean_acc = float(np.mean(list(results.values())))
    print(f"\nMean held-out-participant accuracy: {mean_acc:.3f}")
    print(f"Reference honest LOPO accuracy:       {reference_lopo:.3f}")
    print(f"Resubstitution ceiling (must stay below): {RESUB_CEILING:.3f}")

    failures = []
    if mean_acc >= RESUB_CEILING:
        failures.append(f"mean held-out accuracy {mean_acc:.3f} >= resubstitution ceiling "
                         f"{RESUB_CEILING:.3f} — calibration may be leaking training-subject statistics")
    if abs(mean_acc - reference_lopo) > tolerance:
        failures.append(f"mean held-out accuracy {mean_acc:.3f} is more than {tolerance:.2f} away "
                         f"from the reference LOPO accuracy {reference_lopo:.3f} — calibration may be broken")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nPASS: --subject-norm per-subject calibration generalizes to unseen participants "
          "(held-out accuracy tracks honest LOPO, not resubstitution).")


if __name__ == "__main__":
    main()
