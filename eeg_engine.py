#!/usr/bin/env python3
"""
================================================================================
 eeg_engine.py  —  universal back-end for the EEG -> command prototype
================================================================================
Reads ANY motor-imagery dataset, auto-adapts to it (sampling rate, channels,
classes), extracts canonical features, fits & evaluates models PER DATASET under
Leave-One-Participant-Out, and exports a small "playback" JSON that the HTML
prototype animates -> so the SAME drone/lamp demo works on any dataset.

Per-dataset mode (default): the *pipeline* is universal; each dataset gets its
own fitted model. (Cross-dataset transfer with one shared model is a later step.)

Inputs (auto-detected):
  --csv  file.csv     your feature CSV (cols ending __feat + movement + participant)
  --npz  file.npz     raw epochs: X[n_trials,n_ch,n_samp], y, groups, sfreq[, ch_names]
  --raw  file.edf     EDF/GDF/BDF/.set/.vhdr via MNE (needs events/annotations)
  --moabb BNCI2014_001  public dataset via MOABB (run locally; auto-downloads)

Outputs:
  - prints a per-dataset model-comparison table (your model vs baselines)
  - writes playback_<name>.json  (per-trial pred/confidence/true/participant)

Deps: numpy, scipy, scikit-learn  (+ mne for --raw, + moabb for --moabb).
Lives next to mi_experiment.py (re-uses its CSP/FBCSP/Riemann/self-training code).
================================================================================
"""
import argparse, json, os, re, sys
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
import warnings; warnings.filterwarnings("ignore")
import mi_experiment as M   # re-use proven CSP / FBCSP / Riemann / self-training / band-power
from significance import pairwise_significance

# ---------------------------------------------------------------- channel pick
# "takes the signals it needs": keep motor-cortex (sensorimotor) channels.
SM_RE = re.compile(r'^(FC|CP|C)(z|Z|\d+)$')
def select_sensorimotor(X, ch_names):
    """Keep motor-cortex channels (C*, FC*, CP*); robust to naming like 'C3.','FCz'."""
    if not ch_names: return X, None, "all channels (names unknown)"
    def norm(n): return re.sub(r'[^A-Za-z0-9]','',str(n)).upper()
    pat=re.compile(r'^(FC|CP|C)(Z|\d+)$')
    keep=[i for i,n in enumerate(ch_names) if pat.match(norm(n))]
    if len(keep) < 3:
        return X, ch_names, f"all {len(ch_names)} channels (no standard sensorimotor names)"
    names=[ch_names[i] for i in keep]
    return X[:, keep, :], names, f"{len(keep)} sensorimotor channels: {names}"

# --- literature-backed channel subsets (per device) ---------------------------
# Emotiv EPOC X has NO C3/C4; MI studies use frontal/fronto-central F3,F4,FC5,FC6
# (premotor+motor cortex).  Standard 10-20 montages use central C3/Cz/C4 (+FC/CP).
MOTOR4_EMOTIV   = {'F3','F4','FC5','FC6'}
MOTORWIDE_EMOTIV= {'T7','P3','FC5','F3','AF3','AF4','F4','FC6','P4','T8'}
CENTRAL         = {'C3','C1','CZ','C2','C4','C5','C6','FC3','FC1','FCZ','FC2','FC4','FC5','FC6',
                   'CP3','CP1','CPZ','CP2','CP4'}
CENTRAL_WIDE    = CENTRAL | {'F3','F4','AF3','AF4','T7','T8','P3','P4','FC7','FC8','CP5','CP6'}
def _norm(n): return re.sub(r'[^A-Za-z0-9]','',str(n)).upper()
def channels_for(ch_names, mode):
    """Return (kept_indices, mode) for a channel-name list under a subset mode."""
    up=[_norm(n) for n in ch_names]
    has_c=('C3' in up) or ('C4' in up)
    if mode=='all': return list(range(len(ch_names))), 'all'
    want=(CENTRAL if has_c else MOTOR4_EMOTIV) if mode=='motor' \
         else (CENTRAL_WIDE if has_c else MOTORWIDE_EMOTIV)
    keep=[i for i,n in enumerate(up) if n in want]
    return keep, mode

# ---------------------------------------------------------------- loaders
def load_any(args):
    """-> dict(X_epochs, Xfeat, y, groups, sfreq, ch_names, classes, name, raw)"""
    if args.moabb:
        return load_moabb_dataset(args.moabb)
    if args.npz:
        d=np.load(args.npz, allow_pickle=True)
        y=d["y"]; 
        if y.dtype.kind in "US": _,y=np.unique(y,return_inverse=True)
        ch=list(d["ch_names"]) if "ch_names" in d else None
        return dict(X=d["X"] if "X" in d else None,
                    Xfeat=d["Xfeat"] if "Xfeat" in d else None,
                    y=y.astype(int), groups=d["groups"],
                    sfreq=float(d["sfreq"]) if "sfreq" in d else 128.0,
                    ch_names=ch, name=os.path.basename(args.npz), raw=("X" in d))
    if args.raw:
        return load_mne(args.raw)
    if args.csv:
        Xfeat,y,groups,kept,feat_names = load_csv_ch(args.csv, getattr(args,'channels','all'), getattr(args,'asymmetry',False))
        return dict(X=None,Xfeat=Xfeat,y=y,groups=groups,sfreq=128.0,
                    ch_names=None, name=os.path.basename(args.csv), raw=False,
                    kept_channels=kept, feature_names=feat_names)
    sys.exit("provide --csv / --npz / --raw / --moabb")



def hemispheric_pairs(ch_names):
    """Pair channels by standard 10-20 left/right numbering (odd=left, even=right, same prefix)."""
    groups={}
    for n in ch_names:
        m=re.match(r'^([A-Za-z]+)(\d+)$', n)
        if not m: continue
        prefix,num=m.group(1).upper(),int(m.group(2))
        groups.setdefault(prefix,{})[num]=n
    pairs=[]
    for nums in groups.values():
        for num in nums:
            if num%2==1 and (num+1) in nums:
                pairs.append((nums[num], nums[num+1]))
    return pairs

def build_asymmetry_features(df, cols, chan_fn, band_fn):
    """Left-right log-power difference per hemispheric channel pair per band
    (classic MI laterality index: contralateral desync during hand imagery)."""
    by_ch_band={}
    for c in cols:
        ch,bd=chan_fn(c),band_fn(c)
        if ch and bd: by_ch_band[(ch,bd)]=c
    ch_names=sorted(set(k[0] for k in by_ch_band))
    pairs=hemispheric_pairs(ch_names)
    if not pairs: return None
    bands=sorted(set(k[1] for k in by_ch_band))
    feats=[]
    for left,right in pairs:
        for bd in bands:
            lk,rk=(left,bd),(right,bd)
            if lk in by_ch_band and rk in by_ch_band:
                feats.append((df[by_ch_band[lk]]-df[by_ch_band[rk]]).to_numpy(float))
    return np.column_stack(feats) if feats else None

def load_csv_ch(path, mode='all', asymmetry=False):
    """Load a feature CSV; optionally keep only the motor-channel feature columns
    and/or append hemispheric asymmetry (left-right) features.
    Returns (Xfeat, y, groups, kept_channels, feature_names) — feature_names is
    the ordered column list backing Xfeat's columns (needed to persist an exact
    feature spec for model bundles, see train_model.py)."""
    import pandas as pd
    df=pd.read_csv(path)
    feat=[c for c in df.columns if c.endswith('__feat')]
    def chan(col):
        m=re.match(r'eeg\.([A-Za-z]+\d*)', col); return _norm(m.group(1)) if m else None
    def band(col):
        m=re.search(r'_(\d+_\d+)Hz__feat$', col); return m.group(1) if m else None
    chans=[chan(c) for c in feat]; uniq=sorted(set(x for x in chans if x))
    keep_idx,_=channels_for(uniq, mode); keepset=set(uniq[i] for i in keep_idx)
    cols=[c for c,ch in zip(feat,chans) if ch in keepset]
    if not cols: cols=feat; keepset=set(uniq)   # never return empty
    Xfeat=df[cols].to_numpy(float); y=df['movement'].to_numpy()
    if y.dtype.kind in 'US': _,y=np.unique(y,return_inverse=True)
    feature_names=list(cols)
    if asymmetry:
        asym=build_asymmetry_features(df, cols, chan, band)
        if asym is not None:
            Xfeat=np.hstack([Xfeat, asym])
            feature_names=feature_names+[f"asym_{i}" for i in range(asym.shape[1])]
    return Xfeat, y.astype(int), df['participant'].to_numpy(), sorted(keepset), feature_names

def load_moabb_dataset(name):
    """Load a MOABB MI dataset as a clean 2-class problem, WITH channel names
    (so the engine can trim to sensorimotor channels)."""
    import importlib
    from moabb.paradigms import MotorImagery
    mod=importlib.import_module('moabb.datasets')
    def grab(*cands):
        for c in cands:
            if hasattr(mod,c): return getattr(mod,c)
        raise ImportError("MOABB dataset not found: %s. Try `pip install -U moabb`." % (cands,))
    def pull(para, ds):
        try:
            ep, labels, meta = para.get_data(dataset=ds, return_epochs=True)
            try: X = ep.get_data(copy=False)
            except TypeError: X = ep.get_data()
            return X, [str(x) for x in labels], meta, list(ep.ch_names), float(ep.info['sfreq'])
        except Exception:
            X, labels, meta = para.get_data(dataset=ds)
            return X, [str(x) for x in labels], meta, None, float(getattr(para,'resample',None) or 250.0)
    n=name.lower().replace('-','').replace('_','')
    if n in ('2b','bnci2b','bnci2014004'):
        DS=grab('BNCI2014_004','BNCI2014004'); tag='BCI-IV-2b'
        X,labels,meta,ch,sf=pull(MotorImagery(n_classes=2,fmin=0.5,fmax=40,events=['left_hand','right_hand']),DS())
    elif n in ('2a','bnci2a','bnci2014001'):
        DS=grab('BNCI2014_001','BNCI2014001'); tag='BCI-IV-2a (L/R)'
        X,labels,meta,ch,sf=pull(MotorImagery(n_classes=2,fmin=0.5,fmax=40,events=['left_hand','right_hand']),DS())
    elif n in ('physionet','eegbci','physionetmi'):
        DS=grab('PhysionetMI','PhysionetMotorImagery'); tag='PhysioNet-MI'
        pk=dict(fmin=0.5,fmax=40,resample=160.0,tmin=0.0,tmax=3.0)
        X=None; last=None
        for ev in (['left_hand','right_hand'],['hands','feet']):
            try:
                X,labels,meta,ch,sf=pull(MotorImagery(n_classes=2,events=ev,**pk),DS())
                tag='PhysioNet-MI ('+'/'.join(ev)+')'; break
            except Exception as e: last=e
        if X is None: raise last
    else:
        DS=grab(name); tag=name
        X,labels,meta,ch,sf=pull(MotorImagery(n_classes=2,fmin=0.5,fmax=40),DS())
    classnames,y=np.unique(labels, return_inverse=True)
    groups=meta["subject"].to_numpy()
    print("  loaded %s: %d trials, %d subjects, %d channels, classes=%s"
          % (tag,len(y),len(np.unique(groups)),len(ch),list(classnames)))
    return dict(X=X.astype(float), Xfeat=None, y=y.astype(int), groups=groups,
                sfreq=sf, ch_names=ch, name=tag, raw=True,
                classnames=[str(c) for c in classnames])

def load_mne(path):
    """Best-effort epoching of a raw EEG file using its events/annotations."""
    import mne
    ext=os.path.splitext(path)[1].lower()
    rd={'.edf':mne.io.read_raw_edf,'.bdf':mne.io.read_raw_bdf,'.gdf':mne.io.read_raw_gdf,
        '.set':mne.io.read_raw_eeglab,'.vhdr':mne.io.read_raw_brainvision}.get(ext)
    if rd is None: sys.exit(f"unsupported raw type {ext}")
    raw=rd(path, preload=True, verbose=False)
    raw.filter(0.5,40.,verbose=False)
    try: events,event_id=mne.events_from_annotations(raw, verbose=False)
    except Exception: events=mne.find_events(raw, verbose=False); event_id=None
    if events is None or len(events)==0:
        sys.exit("no events/annotations found to epoch — use MOABB for this dataset.")
    ep=mne.Epochs(raw,events,event_id=event_id,tmin=0.0,tmax=2.0,baseline=None,
                  preload=True,verbose=False)
    X=ep.get_data(); y=ep.events[:,-1]
    _,y=np.unique(y,return_inverse=True)
    groups=np.zeros(len(y),int)            # single-subject file -> one group
    return dict(X=X.astype(float),Xfeat=None,y=y.astype(int),groups=groups,
                sfreq=float(raw.info['sfreq']),ch_names=raw.ch_names,
                name=os.path.basename(path), raw=True)

def build_features(D, channels):
    """Shared adapt step: raw epochs -> channel-select + Welch band-power features
    (D['raw']==True), or the already channel-selected feature CSV columns
    (D['raw']==False). Returns (Xfeat, feature_names). Used by both
    eeg_engine.main() and train_model.py/classify.py so the exact feature layout
    persisted in a model bundle always matches what the engine itself would build."""
    if D["raw"]:
        X=np.asarray(D["X"],float); chn=D.get("ch_names")
        if chn:
            keep,_=channels_for(chn, channels)
            if channels!="all" and 0<len(keep)<len(chn):
                names=[chn[i] for i in keep]; X=X[:,keep,:]
                print(f"Channel selection [{channels}]: kept {len(keep)}/{len(chn)} -> {names}")
            else:
                names=chn
                print(f"Channel selection [{channels}]: using all {len(chn)} channels"
                      + ("" if channels=="all" else " (no subset match — kept all)"))
        else:
            names=None
            print("Channel selection: all channels (names unknown)")
        print("Extracting Welch band-power features ...")
        Xfeat=M.bandpower_features(X, D["sfreq"])
        ch_labels=names or [f"ch{i}" for i in range(X.shape[1])]
        feature_names=[f"{ch}_{b}Hz" for ch in ch_labels for b,_,_ in M.BANDS]
    else:
        Xfeat=np.asarray(D["Xfeat"],float)
        feature_names=list(D.get("feature_names") or [f"f{i}" for i in range(Xfeat.shape[1])])
        kc=D.get("kept_channels")
        print(f"Feature matrix: {Xfeat.shape[1]} features"
              + (f" from {len(kc)} channels {kc}" if kc else "")
              + " (no raw signals; CSP/FBCSP/Riemannian need raw epochs).")
    return Xfeat, feature_names

def per_subject_normalize(Xfeat, groups):
    """Z-score each participant's features using only their OWN trials, before
    pooling across subjects for training. Removes each subject's baseline shift
    (skull/impedance/arousal-level differences) that otherwise dominates the
    cross-subject feature distribution; keeps the within-subject class-related
    pattern. Uses no labels, so it's valid to apply to a held-out LOPO subject
    too (mirrors real calibration: a few seconds of the new user's own signal)."""
    Xout=np.empty_like(Xfeat)
    for g in np.unique(groups):
        idx=groups==g
        mu=Xfeat[idx].mean(0); sd=Xfeat[idx].std(0); sd[sd<1e-12]=1.0
        Xout[idx]=(Xfeat[idx]-mu)/sd
    return Xout

# ---------------------------------------------------------------- splitter
def get_splitter(groups):
    return (LeaveOneGroupOut(), "LOPO (leave-one-participant-out)") if len(np.unique(groups))>1 \
           else (StratifiedKFold(5,shuffle=True,random_state=42), "5-fold (single subject)")

# ---------------------------------------------------------------- playback
def make_playback(Xfeat,y,groups,name,classes):
    """Honest cross-validated per-trial predictions using YOUR pipeline (band-power+SVM-RBF)."""
    spl,proto=get_splitter(groups); trials=[None]*len(y)
    for tr,te in spl.split(Xfeat,y,groups):
        sc=StandardScaler().fit(Xfeat[tr])
        clf=SVC(kernel='rbf',C=1.0,gamma='scale',probability=True,random_state=42).fit(sc.transform(Xfeat[tr]),y[tr])
        pr=clf.predict_proba(sc.transform(Xfeat[te])); pred=clf.classes_[pr.argmax(1)]; conf=pr.max(1)
        for k,ti in enumerate(te):
            trials[ti]={"pred":int(pred[k]),"conf":round(float(conf[k]),3),
                        "true":int(y[ti]),"participant":str(groups[ti])}
    acc=float(np.mean([t["pred"]==t["true"] for t in trials]))
    cmap={str(i):("UP" if i==1 else "DOWN") for i in range(len(classes))} if len(classes)==2 \
         else {str(i):c for i,c in enumerate(classes)}
    return {"dataset":name,"protocol":proto,"n_trials":len(trials),
            "classes":list(classes),"command_map":cmap,
            "model":"SVM-RBF on band-power (your pipeline)","accuracy":round(acc,3),
            "trials":trials}


# ---------------------------------------------------------------- model roster
def get_model_builders():
    """The model roster, keyed by short CLI-friendly name -> (full display name,
    zero-arg builder). Single source of truth for run_roster's comparison table,
    lopo_single_model(), and train_model.py's persisted deployment model — so a
    model bundle's honest accuracy is always directly comparable to
    results_master.csv / summary_report.md."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    try:
        from xgboost import XGBClassifier
        def GBM(): return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
        gbm_name="XGBoost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        def GBM(): return HistGradientBoostingClassifier(random_state=42)
        gbm_name="XGBoost/HGBT"
    return {
        "dt":     ("Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
        "rf":     ("Random Forest", lambda: RandomForestClassifier(n_estimators=300, random_state=42)),
        "xgb":    (gbm_name, GBM),
        "lda":    ("Shrinkage-LDA", lambda: LDA(solver="lsqr", shrinkage="auto")),
        "logreg": ("Logistic Regression", lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        "mlp":    ("MLP (64)", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=42)),
        "svm":    ("SVM-RBF (supervised, 100%)", lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    }

def lopo_single_model(Xfeat, y, groups, model_key):
    """Honest cross-subject accuracy for ONE roster model, using the exact same
    per-fold protocol as run_roster (fresh StandardScaler + model fit on the
    train fold only) — reused by train_model.py so a persisted bundle's stored
    accuracy is computed identically to the engine's comparison table, never
    reinvented. Returns (full_name, acc_mean, acc_std, protocol_str)."""
    from sklearn.metrics import accuracy_score
    builders=get_model_builders()
    if model_key not in builders:
        raise ValueError(f"unknown model '{model_key}'; choices: {list(builders)}")
    full_name,mk=builders[model_key]
    spl,proto=get_splitter(groups)
    accs=[]
    for tr,te in spl.split(Xfeat,y,groups):
        sc=StandardScaler().fit(Xfeat[tr])
        clf=mk().fit(sc.transform(Xfeat[tr]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(sc.transform(Xfeat[te]))))
    return full_name, float(np.mean(accs)), float(np.std(accs)), proto

def run_roster(Xfeat, y, groups, label_frac=0.20):
    """YOUR classifier roster on band-power features (same for every dataset)."""
    from sklearn.metrics import accuracy_score, f1_score
    builders={full:mk for full,mk in get_model_builders().values()}
    stn="Self-Training SVM-RBF (%d%% labels)"%int(label_frac*100)
    res={k:{"acc":[],"f1":[]} for k in list(builders)+[stn]}
    spl,_=get_splitter(groups); rng=np.random.RandomState(42)
    fold_ids=[]
    for i,(tr,te) in enumerate(spl.split(Xfeat,y,groups)):
        ytr,yte=y[tr],y[te]
        fold_ids.append(str(groups[te[0]]) if len(np.unique(groups))>1 else f"fold_{i}")
        sc=StandardScaler().fit(Xfeat[tr]); Xtr=sc.transform(Xfeat[tr]); Xte=sc.transform(Xfeat[te])
        for name,mk in builders.items():
            yp=mk().fit(Xtr,ytr).predict(Xte)
            res[name]["acc"].append(accuracy_score(yte,yp)); res[name]["f1"].append(f1_score(yte,yp,average="macro"))
        idx=np.arange(len(tr)); rng.shuffle(idx); k=max(int(round(label_frac*len(tr))),2)
        yp=M.self_training_svm(Xfeat[tr][idx[:k]], ytr[idx[:k]], Xfeat[tr][idx[k:]], Xfeat[te])
        res[stn]["acc"].append(accuracy_score(yte,yp)); res[stn]["f1"].append(f1_score(yte,yp,average="macro"))
    return res, fold_ids


# ---------------------------------------------------------------- results file
def append_results(path, dataset, channels, nparts, ntrials, res):
    """Append one row per model to a master CSV (de-duplicating on dataset+channels+model)."""
    import csv, os
    cols=["dataset","channels","participants","trials","model","acc_mean","acc_std","macro_f1"]
    rows=[]
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f): rows.append(r)
    key=lambda r:(r["dataset"],r["channels"],r["model"])
    newkeys={(dataset,channels,m) for m in res}
    rows=[r for r in rows if key(r) not in newkeys]           # drop stale duplicates
    import numpy as np
    for m,d in res.items():
        rows.append({"dataset":dataset,"channels":channels,"participants":nparts,"trials":ntrials,
                     "model":m,"acc_mean":round(float(np.mean(d["acc"])),3),
                     "acc_std":round(float(np.std(d["acc"])),3),
                     "macro_f1":round(float(np.mean(d["f1"])),3)})
    rows.sort(key=lambda r:(r["dataset"],r["channels"],-float(r["acc_mean"])))
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    return len(rows)

def append_folds(path, dataset, channels, fold_ids, res, protocol):
    """Persist per-fold acc/f1 for every model in this run to a master JSON
    (dedup by dataset|channels, like append_results) so significance testing
    doesn't need to be redone from raw data every time."""
    key=f"{dataset}|{channels}"
    data={}
    if os.path.exists(path):
        with open(path) as f: data=json.load(f)
    data[key]={"dataset":dataset,"channels":channels,"protocol":protocol,
               "fold_ids":list(fold_ids),
               "models":{m:{"acc":[round(float(x),4) for x in d["acc"]],
                            "f1":[round(float(x),4) for x in d["f1"]]} for m,d in res.items()}}
    with open(path,"w") as f: json.dump(data,f,indent=1)
    return len(data)

def print_table_sig(res, sig, title):
    """Like mi_experiment.print_table, but marks the best model with * and any
    model not significantly different from it (Holm-Bonferroni-corrected paired
    Wilcoxon across folds) with ~."""
    print("\n"+"="*72); print(title); print("="*72)
    print(f"{'Model':<38}{'Acc mean':>10}{'Acc std':>10}{'macro-F1':>10}")
    print("-"*72)
    order=sorted(res.items(), key=lambda kv: -np.mean(kv[1]['acc']))
    for name,d in order:
        mark=" *" if sig[name]["is_best"] else (" ~" if sig[name]["tied_with_best"] else "")
        print(f"{name+mark:<38}{np.mean(d['acc']):>10.3f}{np.std(d['acc']):>10.3f}{np.mean(d['f1']):>10.3f}")
    print("-"*72)
    print("(chance = 0.50 for binary;  LOPO = leave-one-participant-out)")
    print("(* best mean accuracy;  ~ not significantly different from best — paired Wilcoxon,")
    print(" Holm-Bonferroni corrected across all pairs, alpha=0.05)")

def append_log(path, dataset, channels, nparts, ntrials, res, sig):
    """Append a timestamped run summary to the RESULTS_LOG.md lab notebook."""
    import datetime
    best=[m for m,d in sig.items() if d["is_best"]][0]
    tied=sorted(m for m,d in sig.items() if d["tied_with_best"] and not d["is_best"])
    ts=datetime.datetime.now().isoformat(timespec="seconds")
    lines=[f"\n## {ts} — {dataset} [{channels}]",
           f"- participants={nparts}, trials={ntrials}",
           f"- best: **{best}** (acc={np.mean(res[best]['acc']):.3f}, macro-F1={np.mean(res[best]['f1']):.3f})",
           f"- tied with best (Holm-Bonferroni, alpha=0.05): {', '.join(tied) if tied else '(none)'}",
           "- full table: results_master.csv / folds_master.json\n"]
    with open(path,"a") as f: f.write("\n".join(lines))

# ---------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser(description="Universal EEG->command engine (per-dataset mode).")
    ap.add_argument("--csv"); ap.add_argument("--npz"); ap.add_argument("--raw"); ap.add_argument("--moabb"); ap.add_argument("--dataset", dest="moabb")
    ap.add_argument("--playback", help="output playback json (default playback_<name>.json)")
    ap.add_argument("--no-compare", action="store_true", help="skip baseline comparison (fast)")
    ap.add_argument("--label-frac", type=float, default=0.20)
    ap.add_argument("--channels", choices=["all","motor","motor-wide"], default="all",
                    help="channel subset: all | motor (Emotiv:F3/F4/FC5/FC6 · 10-20:C3/Cz/C4) | motor-wide")
    ap.add_argument("--asymmetry", action="store_true",
                    help="append left-right hemispheric band-power difference features (CSV mode only)")
    ap.add_argument("--subject-norm", dest="subject_norm", action="store_true",
                    help="z-score each participant's features using only their own trials, before pooling for training")
    ap.add_argument("--save-table", dest="save_table", default="results_master.csv",
                    help="master CSV to append each run's results to (default results_master.csv)")
    ap.add_argument("--folds-table", dest="folds_table", default="folds_master.json",
                    help="master JSON of per-fold acc/f1 per model, for significance testing (default folds_master.json)")
    ap.add_argument("--log", dest="log_path", default="RESULTS_LOG.md",
                    help="lab-notebook markdown log to append each run's summary to (default RESULTS_LOG.md)")
    a=ap.parse_args()

    D=load_any(a); name=re.sub(r'\W+','_',D["name"]).strip('_')
    classes=[str(c) for c in np.unique(D["y"])]
    if len(classes)==2: classes=["Down","Up"]
    print(f"\nDataset: {D['name']}  |  trials={len(D['y'])}  |  classes={list(np.unique(D['y']))}  "
          f"|  participants={len(np.unique(D['groups']))}  |  sfreq={D['sfreq']}")

    # ---- adapt: pick signals + build canonical band-power features ----
    Xfeat, _feature_names = build_features(D, a.channels)

    y=np.asarray(D["y"]); groups=np.asarray(D["groups"])
    if getattr(a,'subject_norm',False):
        Xfeat=per_subject_normalize(Xfeat, groups)
        print("Applied per-subject normalization (z-score each participant using their own trials).")
    chlabel = a.channels + ("+asym" if getattr(a,'asymmetry',False) else "") \
                         + ("+subjnorm" if getattr(a,'subject_norm',False) else "")

    # ---- per-dataset model comparison (your model vs literature baselines) ----
    if not a.no_compare:
        res,fold_ids=run_roster(Xfeat, y, groups, label_frac=a.label_frac)
        sig=pairwise_significance({m:d["acc"] for m,d in res.items()})
        print_table_sig(res, sig, f"PER-DATASET comparison on {D['name']}  [channels={chlabel}, band-power features]")
        _,proto=get_splitter(groups)
        if a.save_table:
            n=append_results(a.save_table, D["name"], chlabel,
                             int(len(np.unique(groups))), int(len(y)), res)
            print(f"results appended -> {a.save_table}  ({n} rows total across all datasets/runs)")
        if a.folds_table:
            append_folds(a.folds_table, D["name"], chlabel, fold_ids, res, proto)
            print(f"per-fold results appended -> {a.folds_table}")
        if a.log_path:
            append_log(a.log_path, D["name"], chlabel, int(len(np.unique(groups))), int(len(y)), res, sig)
            print(f"run summary appended -> {a.log_path}")

    # ---- playback for the prototype ----
    pb=make_playback(Xfeat,y,groups,D["name"],classes)
    suffix = "" if chlabel=="all" else "_"+chlabel.replace("+","_")
    out=a.playback or f"playback_{name}{suffix}.json"
    json.dump(pb, open(out,"w"))
    print(f"\nplayback written -> {out}   (accuracy {pb['accuracy']}, {pb['n_trials']} trials)")
    print("Load this file in the prototype (button: 'Load playback') to animate this dataset.")

if __name__=="__main__":
    main()
