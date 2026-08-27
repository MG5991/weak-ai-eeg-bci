#!/usr/bin/env python3
# ============================================================
#  ONE-CLICK LAUNCHER  —  just press ▶ Run in VS Code / PyCharm
# ============================================================
# Runs the universal engine on the dataset named below and writes a
# playback_*.json that you then load in EEG_control_prototype.html.
import os, sys

# >>> change this to any CSV sitting in this same folder <<<
DATASET = "eeg_dataset_emotiv.csv"

here = os.path.dirname(os.path.abspath(__file__))
path = DATASET if os.path.isabs(DATASET) else os.path.join(here, DATASET)
if not os.path.exists(path):
    print("[!] Dataset not found:", path)
    print("    Put your CSV in this folder, or edit DATASET at the top of run_me.py.")
    sys.exit(1)

try:
    import eeg_engine
except ImportError:
    print("[!] Keep run_me.py, eeg_engine.py and mi_experiment.py in the SAME folder.")
    sys.exit(1)

sys.argv = ["eeg_engine.py", "--csv", path, "--subject-norm"]
eeg_engine.main()
print("\nNEXT: open EEG_control_prototype.html in your browser →"
      " 'Load playback' → choose the playback_*.json just created.")
