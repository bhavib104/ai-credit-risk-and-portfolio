import os, json, uuid, datetime, joblib
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss, confusion_matrix
from ml.data import generate
from ml.features import engineer, build_preprocessor, feature_columns, SENSITIVE
from backend.config import settings
from backend.db import SessionLocal, ModelRegistry, ModelEvaluation

RANDOM_SEED = 42

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def train():
    df = generate()
    df = engineer(df)
    X = df[feature_columns()]
    y = df["TARGET"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RANDOM_SEED)

    pre = build_preprocessor()
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1),
    }
    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", model)])
        pipe.fit(Xtr, ytr)
        val_prob = pipe.predict_proba(Xval)[:, 1]
        results[name] = evaluate(yval, val_prob)
        fitted[name] = pipe

    # select by pr_auc
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_pipe = fitted[best_name]
    # calibrate
    cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)

    test_prob = cal.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(yte, test_prob)

    # save artifact
    os.makedirs(settings.artifacts_dir, exist_ok=True)
    version = "v1.0.0"
    art_dir = os.path.join(settings.artifacts_dir, version)
    os.makedirs(art_dir, exist_ok=True)
    joblib.dump(cal, os.path.join(art_dir, "model.joblib"))
    joblib.dump(feature_columns(), os.path.join(art_dir, "features.joblib"))
    with open(os.path.join(art_dir, "metrics.json"), "w") as f:
        json.dump({"validation": results, "test": test_metrics, "best": best_name}, f, indent=2)

    # register
    db = SessionLocal()
    db.add(ModelRegistry(model_id=str(uuid.uuid4()), name=best_name, version=version,
                         dataset_ref="synthetic_credit_v1", metrics=test_metrics, status="deployed"))
    db.add(ModelEvaluation(eval_id=str(uuid.uuid4()), model_version=version, split_name="test",
                           metrics=test_metrics))
    db.commit()
    db.close()

    return {"best": best_name, "version": version, "test_metrics": test_metrics, "all": results}

if __name__ == "__main__":
    from backend.db import init_db
    init_db()
    print(train())
