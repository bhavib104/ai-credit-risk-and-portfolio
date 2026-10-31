import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

def psi(expected, actual, bins=10):
    e = pd.qcut(expected, bins, duplicates="drop")
    a = pd.cut(actual, bins=e.cat.categories, include_lowest=True)
    e_pct = e.value_counts(normalize=True).sort_index()
    a_pct = a.value_counts(normalize=True).reindex(e_pct.index).fillna(1e-6)
    psi_val = ((a_pct - e_pct) * np.log((a_pct + 1e-6) / (e_pct + 1e-6))).sum()
    return float(psi_val)

def ks_drift(expected, actual):
    stat, p = ks_2samp(expected, actual)
    return {"statistic": float(stat), "p_value": float(p), "drift": p < 0.05}

def prediction_drift(expected_probs, actual_probs):
    return {"psi": psi(expected_probs, actual_probs), "ks": ks_drift(expected_probs, actual_probs)}
