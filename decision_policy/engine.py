import yaml
from pathlib import Path

POLICY_VERSION = "1.0"
CONFIG = Path("configs/thresholds.yaml")

def load_policy():
    with open(CONFIG) as f:
        return yaml.safe_load(f)

def classify(pd_val):
    cfg = load_policy()
    cats = cfg["risk_categories"]
    for cat, (lo, hi) in cats.items():
        if lo <= pd_val <= hi or (cat == "high" and pd_val >= lo):
            return cat
    return "high"

def recommend(pd_val):
    cat = classify(pd_val)
    cfg = load_policy()
    rec = cfg["decision_policy"][cat]
    return cat, rec, POLICY_VERSION
