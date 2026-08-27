#!/usr/bin/env python3
"""
Shared statistical-significance helpers for eeg_engine.py and summarize.py.

Paired Wilcoxon signed-rank test between each pair of models in a roster,
evaluated on the SAME per-fold accuracy arrays (same LOPO/K-fold splits for
every model in a given run), with Holm-Bonferroni correction across all
pairwise comparisons in that roster (controls family-wise error rate; with
8 models that's 28 pairs, so an uncorrected alpha=0.05 would over-claim
"different" results by chance alone).
"""
from itertools import combinations
import numpy as np
from scipy.stats import wilcoxon

def holm_bonferroni(pvals, alpha=0.05):
    """pvals: list of p-values (any order) -> list of booleans (reject H0, i.e. significant),
    same order as input. Standard step-down Holm procedure."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    reject = [False]*n
    for rank, idx in enumerate(order):
        thresh = alpha/(n-rank)
        if pvals[idx] <= thresh:
            reject[idx] = True
        else:
            break   # step-down: stop at first non-rejection
    return reject

def _pair_pvalue(acc_a, acc_b):
    """Paired Wilcoxon signed-rank p-value; 1.0 (can't reject / treat as tied) if degenerate
    (identical arrays, or too few non-zero paired differences for the test to run)."""
    a, b = np.asarray(acc_a), np.asarray(acc_b)
    if np.allclose(a, b):
        return 1.0
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        return 1.0   # e.g. all-but-one differences are zero

def pairwise_significance(model_accs, alpha=0.05):
    """model_accs: dict model_name -> list/array of per-fold accuracy (same fold order
    for every model). Returns dict model_name -> {"mean_acc","is_best","tied_with_best","p_vs_best"}.
    "tied_with_best": True if not significantly different from the best model after
    Holm-Bonferroni correction across all C(n,2) pairs in this roster."""
    names = list(model_accs)
    means = {m: float(np.mean(model_accs[m])) for m in names}
    best = max(names, key=lambda m: means[m])
    pairs = list(combinations(names, 2))
    pvals = [_pair_pvalue(model_accs[a], model_accs[b]) for a, b in pairs]
    reject = holm_bonferroni(pvals, alpha=alpha)
    sig_diff = {}   # frozenset({a,b}) -> True if significantly different
    for (a, b), r in zip(pairs, reject):
        sig_diff[frozenset((a, b))] = r
    out = {}
    for m in names:
        is_best = (m == best)
        if is_best:
            tied = True; p_vs_best = 1.0
        else:
            key = frozenset((m, best))
            diff = sig_diff.get(key, False)
            tied = not diff
            p_vs_best = pvals[pairs.index((m, best))] if (m, best) in pairs else \
                        pvals[pairs.index((best, m))]
        out[m] = {"mean_acc": means[m], "is_best": is_best,
                  "tied_with_best": tied, "p_vs_best": p_vs_best}
    return out
