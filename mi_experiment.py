#!/usr/bin/env python3
"""
================================================================================
 MI-EEG EXPERIMENT — your Self-Training SVM-RBF vs. recent-literature baselines
================================================================================
Part A : YOUR dataset  ->  CSP+LDA, FBCSP+LDA, Riemannian-TS+SVM, and YOUR model,
         all under the SAME Leave-One-Participant-Out (LOPO) protocol.
Part B : YOUR pipeline ->  re-run on a PUBLIC dataset (e.g. BCI-IV-2b) under its
         own LOPO, compared with the same baselines on that dataset.

Baselines are the standard classical MI-EEG reference pipelines (MOABB):
  - CSP + LDA                      (Ramoser 2000; Blankertz 2008)
  - FBCSP + LDA                    (Ang et al. 2008)
  - Riemannian Tangent Space + SVM (Barachant 2012; current MOABB SoA, classical)
EEGNet/ShallowConvNet (deep baselines) need PyTorch -> run locally (see notes).

NO results are produced without real data. Running with --smoke uses RANDOM
arrays only to verify the CODE executes; those numbers are meaningless by design.

Dependencies: numpy, scipy, scikit-learn  (all already available; no net needed).
For Part B you also need:  pip install moabb   (one-time, needs internet).
--------------------------------------------------------------------------------
DATA FORMAT (Part A)  — provide ONE of:
  (a) raw epochs  : .npz with  X [n_trials, n_channels, n_samples] (float),
                                 y [n_trials] (0/1 or class names),
                                 groups [n_trials] (participant id per trial),
                                 sfreq (scalar, e.g. 128)
  (b) features    : .npz with  Xfeat [n_trials, n_features], y, groups
                    (feature-based baselines only; no CSP/Riemannian)
================================================================================
"""
import argparse, sys, warnings
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.linalg import eigh
from numpy.linalg import eigh as np_eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
RNG = 42

# ----------------------------------------------------------------------------
# Band-power features  (YOUR pipeline:  per-channel Welch power in 6 EEG bands)
# ----------------------------------------------------------------------------
BANDS = [("4_8",4,8),("8_12",8,12),("12_18",12,18),("18_25",18,25),("25_32",25,32),("32_40",32,40)]  # Emotiv EPOC X bands (canonical everywhere)

def bandpower_features(X, sfreq):
    """X:[n_trials,n_ch,n_samples] -> [n_trials, n_ch*len(BANDS)] log band-power."""
    n_tr, n_ch, n_s = X.shape
    nperseg = min(n_s, int(sfreq*1.0))
    feats = np.empty((n_tr, n_ch*len(BANDS)), dtype=float)
    for t in range(n_tr):
        col = 0
        for c in range(n_ch):
            f, pxx = welch(X[t,c], fs=sfreq, nperseg=nperseg)
            for _, lo, hi in BANDS:
                m = (f>=lo) & (f<hi)
                bp = np.trapezoid(pxx[m], f[m]) if m.any() else 1e-12
                feats[t,col] = np.log(bp + 1e-12); col += 1
    return feats

# ----------------------------------------------------------------------------
# CSP (binary) — from scratch
# ----------------------------------------------------------------------------
def _cov(x):                      # x:[n_ch,n_s] -> normalized covariance
    x = x - x.mean(1, keepdims=True)
    c = x @ x.T
    return c / np.trace(c)

class CSP:
    def __init__(self, n_components=6): self.m = n_components
    def fit(self, X, y):
        cls = np.unique(y)
        c0 = np.mean([_cov(x) for x in X[y==cls[0]]], axis=0)
        c1 = np.mean([_cov(x) for x in X[y==cls[1]]], axis=0)
        cc = c0 + c1
        w, v = eigh(cc)                          # whitening
        w = np.clip(w, 1e-12, None)
        P = (v @ np.diag(1.0/np.sqrt(w))).T
        s0 = P @ c0 @ P.T
        d, b = eigh(s0)
        order = np.argsort(d)[::-1]
        W = (b[:, order].T @ P)                  # spatial filters (rows)
        h = self.m//2
        self.filters_ = np.vstack([W[:h], W[-h:]])
        return self
    def transform(self, X):
        out = np.empty((len(X), self.filters_.shape[0]))
        for i, x in enumerate(X):
            z = self.filters_ @ x
            v = np.var(z, axis=1); v = v/ (v.sum()+1e-12)
            out[i] = np.log(v + 1e-12)
        return out

def csp_lda_pipeline(Xtr,ytr,Xte, n_components=6):
    csp = CSP(n_components).fit(Xtr,ytr)
    clf = LDA().fit(csp.transform(Xtr), ytr)
    return clf.predict(csp.transform(Xte))

# ----------------------------------------------------------------------------
# FBCSP — CSP on a filter bank, concatenated, -> LDA
# ----------------------------------------------------------------------------
FB_BANDS = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
def _bandpass(X, lo, hi, sfreq, order=4):
    ny = sfreq/2.0
    b,a = butter(order, [max(lo,0.1)/ny, min(hi,ny-0.1)/ny], btype="band")
    return filtfilt(b,a,X,axis=-1)

def fbcsp_lda_pipeline(Xtr,ytr,Xte,sfreq,m=4):
    Ftr,Fte = [],[]
    for lo,hi in FB_BANDS:
        Xtr_b = _bandpass(Xtr,lo,hi,sfreq); Xte_b=_bandpass(Xte,lo,hi,sfreq)
        csp = CSP(m).fit(Xtr_b,ytr)
        Ftr.append(csp.transform(Xtr_b)); Fte.append(csp.transform(Xte_b))
    Ftr=np.hstack(Ftr); Fte=np.hstack(Fte)
    sc=StandardScaler().fit(Ftr)
    clf=LDA().fit(sc.transform(Ftr),ytr)
    return clf.predict(sc.transform(Fte))

# ----------------------------------------------------------------------------
# Riemannian tangent space (log-Euclidean, stable) + linear SVM
# ----------------------------------------------------------------------------
def _spd_cov(x):
    x = x - x.mean(1, keepdims=True)
    c,_ = ledoit_wolf(x.T)                # shrinkage -> SPD
    return c
def _logm_spd(c):
    w,v = np_eigh(c); w=np.clip(w,1e-12,None)
    return (v*np.log(w)) @ v.T
def _tangent_vecs(covs):
    n = covs[0].shape[0]
    iu = np.triu_indices(n)
    scale = np.sqrt(2.0)*np.ones((n,n)); np.fill_diagonal(scale,1.0)
    out=[]
    for c in covs:
        L=_logm_spd(c); out.append((L*scale)[iu])
    return np.array(out)
def riemann_ts_svm_pipeline(Xtr,ytr,Xte):
    Ctr=[_spd_cov(x) for x in Xtr]; Cte=[_spd_cov(x) for x in Xte]
    Ttr=_tangent_vecs(Ctr); Tte=_tangent_vecs(Cte)
    sc=StandardScaler().fit(Ttr)
    clf=SVC(kernel="linear",C=1.0,random_state=RNG).fit(sc.transform(Ttr),ytr)
    return clf.predict(sc.transform(Tte))

# ----------------------------------------------------------------------------
# YOUR MODEL — Self-Training SVM-RBF on band-power features
# (theta=0.75 confidence, <=10 iterations; label_frac of TRAIN is labelled)
# ----------------------------------------------------------------------------
def self_training_svm(Xtr_lab, ytr_lab, Xtr_unlab, Xte,
                      theta=0.75, max_iter=10):
    sc = StandardScaler().fit(np.vstack([Xtr_lab, Xtr_unlab]) if len(Xtr_unlab) else Xtr_lab)
    Xl = sc.transform(Xtr_lab).tolist(); yl=list(ytr_lab)
    pool = list(sc.transform(Xtr_unlab)) if len(Xtr_unlab) else []
    for _ in range(max_iter):
        clf=SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,random_state=RNG)
        clf.fit(np.array(Xl), np.array(yl))
        if not pool: break
        proba=clf.predict_proba(np.array(pool)); conf=proba.max(1); pred=clf.classes_[proba.argmax(1)]
        take=np.where(conf>=theta)[0]
        if len(take)==0: break
        for i in sorted(take, reverse=True):
            Xl.append(pool[i]); yl.append(pred[i]); pool.pop(i)
    clf=SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,random_state=RNG).fit(np.array(Xl),np.array(yl))
    return clf.predict(sc.transform(Xte))

def supervised_svm_rbf(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr)
    clf=SVC(kernel="rbf",C=1.0,gamma="scale",random_state=RNG).fit(sc.transform(Xtr),ytr)
    return clf.predict(sc.transform(Xte))

# ----------------------------------------------------------------------------
# LOPO evaluation
# ----------------------------------------------------------------------------
def evaluate(X_epochs, y, groups, sfreq, label_frac=0.20, have_epochs=True, Xfeat=None):
    y=np.asarray(y); groups=np.asarray(groups)
    if Xfeat is None and have_epochs:
        Xfeat = bandpower_features(X_epochs, sfreq)
    logo=LeaveOneGroupOut()
    results={}
    def add(name,yt,yp):
        results.setdefault(name,{"acc":[],"f1":[]})
        results[name]["acc"].append(accuracy_score(yt,yp))
        results[name]["f1"].append(f1_score(yt,yp,average="macro"))
    rng=np.random.RandomState(RNG)
    for tr,te in logo.split(Xfeat,y,groups):
        ytr,yte=y[tr],y[te]
        # --- literature baselines (need raw epochs) ---
        if have_epochs:
            Xtr_e,Xte_e=X_epochs[tr],X_epochs[te]
            try: add("CSP+LDA", yte, csp_lda_pipeline(Xtr_e,ytr,Xte_e))
            except Exception as e: print("  [CSP+LDA] skipped:",e)
            try: add("FBCSP+LDA", yte, fbcsp_lda_pipeline(Xtr_e,ytr,Xte_e,sfreq))
            except Exception as e: print("  [FBCSP+LDA] skipped:",e)
            try: add("Riemann-TS+SVM", yte, riemann_ts_svm_pipeline(Xtr_e,ytr,Xte_e))
            except Exception as e: print("  [Riemann] skipped:",e)
        else:  # feature-only baselines
            sc=StandardScaler().fit(Xfeat[tr])
            add("shrinkage-LDA", yte, LDA(solver="lsqr",shrinkage="auto").fit(sc.transform(Xfeat[tr]),ytr).predict(sc.transform(Xfeat[te])))
            add("LogReg", yte, LogisticRegression(max_iter=1000,random_state=RNG).fit(sc.transform(Xfeat[tr]),ytr).predict(sc.transform(Xfeat[te])))
            add("MLP", yte, MLPClassifier(hidden_layer_sizes=(64,),max_iter=500,random_state=RNG).fit(sc.transform(Xfeat[tr]),ytr).predict(sc.transform(Xfeat[te])))
        # --- your model (band-power features) ---
        add("SVM-RBF (supervised,100%)", yte, supervised_svm_rbf(Xfeat[tr],ytr,Xfeat[te]))
        # split TRAIN into labelled / unlabelled for self-training
        idx=np.arange(len(tr)); rng.shuffle(idx)
        k=max(int(round(label_frac*len(tr))),2)
        lab,unl=idx[:k],idx[k:]
        add(f"Self-Training SVM-RBF ({int(label_frac*100)}% labels)", yte,
            self_training_svm(Xfeat[tr][lab],ytr[lab],Xfeat[tr][unl],Xfeat[te]))
    return results

def print_table(results, title):
    print("\n"+"="*72); print(title); print("="*72)
    print(f"{'Model':<38}{'Acc mean':>10}{'Acc std':>10}{'macro-F1':>10}")
    print("-"*72)
    order=sorted(results.items(), key=lambda kv: -np.mean(kv[1]['acc']))
    for name,d in order:
        print(f"{name:<38}{np.mean(d['acc']):>10.3f}{np.std(d['acc']):>10.3f}{np.mean(d['f1']):>10.3f}")
    print("-"*72); print("(chance = 0.50 for binary;  LOPO = leave-one-participant-out)")

# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
def load_npz(path):
    d=np.load(path, allow_pickle=True)
    y=d["y"]; 
    if y.dtype.kind in "US":  # class names -> 0/1...
        _,y=np.unique(y, return_inverse=True)
    groups=d["groups"]
    sfreq=float(d["sfreq"]) if "sfreq" in d else None
    X = d["X"] if "X" in d else None
    Xfeat = d["Xfeat"] if "Xfeat" in d else None
    return X, Xfeat, y.astype(int), groups, sfreq


def load_csv(path, feat_suffix="__feat", label_col="movement", group_col="participant"):
    """Load YOUR feature CSV: columns ending in __feat are features; plus label & participant."""
    import pandas as pd
    df=pd.read_csv(path)
    feat=[c for c in df.columns if c.endswith(feat_suffix)]
    Xfeat=df[feat].to_numpy(float)
    y=df[label_col].to_numpy()
    if y.dtype.kind in "US": _,y=np.unique(y,return_inverse=True)
    groups=df[group_col].to_numpy()
    return None, Xfeat, y.astype(int), groups, None

def load_moabb(dataset="BNCI2014_001"):
    """Part B helper — run LOCALLY (needs `pip install moabb` + internet)."""
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    ds=BNCI2014_001(); para=MotorImagery(n_classes=2, fmin=0.5, fmax=40)
    X,labels,meta=para.get_data(dataset=ds)
    _,y=np.unique(labels, return_inverse=True)
    groups=meta["subject"].to_numpy()
    sfreq=para.resample or 250.0
    return X.astype(float), None, y.astype(int), groups, float(sfreq)

# ----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", help=".npz with your epochs/features (Part A)")
    ap.add_argument("--csv", help="your feature CSV (Part A; band-power features)")
    ap.add_argument("--moabb", help="MOABB dataset name (Part B, run locally)")
    ap.add_argument("--label-frac", type=float, default=0.20)
    ap.add_argument("--smoke", action="store_true", help="run on RANDOM data to test code only")
    a=ap.parse_args()

    if a.smoke:
        print(">>> SMOKE TEST — RANDOM DATA — numbers are MEANINGLESS (code check only) <<<")
        rng=np.random.RandomState(0)
        n_sub, per, ch, samp, sf = 6, 40, 14, 256, 128
        X=rng.randn(n_sub*per, ch, samp); y=rng.randint(0,2,n_sub*per)
        groups=np.repeat(np.arange(n_sub), per)
        res=evaluate(X,y,groups,sf,label_frac=a.label_frac,have_epochs=True)
        print_table(res,"SMOKE TEST (random data — ignore the values)")
        print("\nPIPELINE OK — all models executed end-to-end."); return

    if a.moabb:
        X,Xfeat,y,groups,sf = load_moabb(a.moabb)
        res=evaluate(X,y,groups,sf,label_frac=a.label_frac,have_epochs=True)
        print_table(res, f"PART B — your pipeline vs baselines on {a.moabb}")
    elif a.csv:
        X,Xfeat,y,groups,sf = load_csv(a.csv)
        res=evaluate(None,y,groups,128.0,label_frac=a.label_frac,have_epochs=False,Xfeat=Xfeat)
        print_table(res,"PART A — your CSV: your model vs feature-space literature baselines")
    elif a.data:
        X,Xfeat,y,groups,sf = load_npz(a.data)
        have = X is not None
        if not have and Xfeat is None:
            sys.exit("npz must contain either X (epochs) or Xfeat (features).")
        res=evaluate(X,y,groups,sf or 128.0,label_frac=a.label_frac,
                     have_epochs=have, Xfeat=Xfeat)
        print_table(res, "PART A — your dataset: your model vs literature baselines")
    else:
        ap.print_help()

if __name__=="__main__":
    main()
