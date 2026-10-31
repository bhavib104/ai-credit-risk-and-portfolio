import os, joblib
import numpy as np
import pandas as pd
import shap
from ml.predict import predict_one
from ml.features import engineer, feature_columns

def shap_local(payload, version="v1.0.0"):
    model = joblib.load(os.path.join(__import__("backend.config", fromlist=["settings"]).settings.artifacts_dir, version, "model.joblib"))
    # build df
    from ml.predict import predict_one as _p
    df = pd.DataFrame([payload])
    # reuse engineer via predict path by reconstructing minimal
    # We'll engineer manually using features.engineer after filling
    for k, v in {
        "DAYS_BIRTH": -(payload.get("age", 30) * 365),
        "DAYS_EMPLOYED": -(payload.get("employment_years", 5) * 365),
        "CNT_CHILDREN": payload.get("dependents", 0),
        "NAME_CONTRACT_TYPE": payload.get("loan_type", "cash_loan"),
        "FLAG_OWN_CAR": payload.get("has_car", 0),
        "FLAG_OWN_REALTY": payload.get("has_realty", 0),
        "CODE_GENDER": payload.get("gender", "M"),
        "AMT_INCOME_TOTAL": payload.get("income", 50000),
        "AMT_CREDIT": payload.get("loan_amount", 10000),
        "num_open_accounts": payload.get("num_open_accounts", 4),
        "num_times_late": payload.get("num_times_late", 0),
        "credit_history_years": payload.get("credit_history_years", 5),
        "loan_term": payload.get("loan_term", 36),
        "interest_rate": payload.get("interest_rate", 10),
    }.items():
        if k not in df.columns:
            df[k] = v
    df = engineer(df)
    X = df[feature_columns()]
    # unwrap calibrated model
    base = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    pre = base.named_steps["pre"]
    clf = base.named_steps["clf"]
    Xt = pre.transform(X)
    explainer = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") else shap.LinearExplainer(clf, Xt)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    vals = sv[0].tolist()
    feature_names = pre.get_feature_names_out().tolist()
    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
    return {
        "shap_values": dict(pairs[:15]),
        "top_increasing": [p for p in pairs if p[1] > 0][:5],
        "top_decreasing": [p for p in pairs if p[1] < 0][:5],
    }
