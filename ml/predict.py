import os, joblib, uuid, datetime
import numpy as np
import pandas as pd
from ml.features import engineer, feature_columns
from backend.config import settings

def load_model(version="v1.0.0"):
    path = os.path.join(settings.artifacts_dir, version, "model.joblib")
    return joblib.load(path)

def predict_one(payload: dict, version="v1.0.0"):
    model = load_model(version)
    df = pd.DataFrame([payload])
    # ensure required engineered fields
    if "DAYS_BIRTH" not in df.columns:
        df["DAYS_BIRTH"] = -(payload.get("age", 30) * 365)
    if "DAYS_EMPLOYED" not in df.columns:
        df["DAYS_EMPLOYED"] = -(payload.get("employment_years", 5) * 365)
    if "CNT_CHILDREN" not in df.columns:
        df["CNT_CHILDREN"] = payload.get("dependents", 0)
    if "NAME_CONTRACT_TYPE" not in df.columns:
        df["NAME_CONTRACT_TYPE"] = payload.get("loan_type", "cash_loan")
    if "FLAG_OWN_CAR" not in df.columns:
        df["FLAG_OWN_CAR"] = payload.get("has_car", 0)
    if "FLAG_OWN_REALTY" not in df.columns:
        df["FLAG_OWN_REALTY"] = payload.get("has_realty", 0)
    if "CODE_GENDER" not in df.columns:
        df["CODE_GENDER"] = payload.get("gender", "M")
    if "AMT_INCOME_TOTAL" not in df.columns:
        df["AMT_INCOME_TOTAL"] = payload.get("income", 50000)
    if "AMT_CREDIT" not in df.columns:
        df["AMT_CREDIT"] = payload.get("loan_amount", 10000)
    if "num_open_accounts" not in df.columns:
        df["num_open_accounts"] = payload.get("num_open_accounts", 4)
    if "num_times_late" not in df.columns:
        df["num_times_late"] = payload.get("num_times_late", 0)
    if "credit_history_years" not in df.columns:
        df["credit_history_years"] = payload.get("credit_history_years", 5)
    if "loan_term" not in df.columns:
        df["loan_term"] = payload.get("loan_term", 36)
    if "interest_rate" not in df.columns:
        df["interest_rate"] = payload.get("interest_rate", 10)
    df = engineer(df)
    X = df[feature_columns()]
    prob = float(model.predict_proba(X)[:, 1][0])
    return prob, version

def risk_category(pd_val, thresholds=None):
    if thresholds is None:
        thresholds = [(0.0, 0.2, "low"), (0.2, 0.5, "medium"), (0.5, 1.0, "high")]
    for lo, hi, cat in thresholds:
        if lo <= pd_val < hi or (cat == "high" and pd_val >= lo):
            return cat
    return "high"
