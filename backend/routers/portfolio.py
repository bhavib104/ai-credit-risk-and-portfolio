from fastapi import APIRouter
from backend.db import SessionLocal, Prediction, LoanApplication
from portfolio.analytics import summary
import pandas as pd

router = APIRouter()

@router.get("/portfolio/summary")
def portfolio_summary():
    db = SessionLocal()
    preds = db.query(Prediction).all()
    apps = {a.application_id: a for a in db.query(LoanApplication).all()}
    db.close()
    rows = []
    for p in preds:
        a = apps.get(p.application_id)
        rows.append({"application_id": p.application_id, "pd": p.pd, "risk_category": p.risk_category,
                     "AMT_CREDIT": a.loan_amount if a else 0, "NAME_CONTRACT_TYPE": a.loan_type if a else "unknown",
                     "AMT_INCOME_TOTAL": 0})
    df = pd.DataFrame(rows)
    if "risk_category" not in df.columns:
        df["risk_category"] = "unknown"
    return summary(df)
