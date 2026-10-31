from fastapi import APIRouter
from pydantic import BaseModel, Field
from ml.predict import predict_one, risk_category
from ml.explain import shap_local
from decision_policy.engine import recommend
import uuid, datetime
from backend.db import SessionLocal, Prediction, LoanApplication, Applicant

router = APIRouter()

class BorrowerInput(BaseModel):
    age: int = Field(35, ge=18, le=100)
    income: float = Field(50000, gt=0)
    employment_years: float = Field(5, ge=0)
    credit_history_years: float = Field(5, ge=0)
    num_open_accounts: int = Field(4, ge=0)
    num_times_late: int = Field(0, ge=0)
    loan_amount: float = Field(10000, gt=0)
    loan_type: str = "cash_loan"
    loan_term: int = 36
    interest_rate: float = 10.0
    dependents: int = 0
    has_car: int = 0
    has_realty: int = 0
    gender: str = "M"

@router.post("/predict")
def predict(b: BorrowerInput):
    prob, ver = predict_one(b.dict())
    cat = risk_category(prob)
    rcat, rec, pver = recommend(prob)
    # persist
    db = SessionLocal()
    app_id = str(uuid.uuid4())
    db.add(Applicant(applicant_id=app_id, age=b.age, income=b.income, employment_years=b.employment_years,
                     credit_history_years=b.credit_history_years, num_open_accounts=b.num_open_accounts,
                     num_times_late=b.num_times_late))
    db.add(LoanApplication(application_id=app_id, applicant_id=app_id, loan_amount=b.loan_amount,
                            loan_type=b.loan_type, loan_term=b.loan_term, interest_rate=b.interest_rate))
    db.add(Prediction(prediction_id=str(uuid.uuid4()), application_id=app_id, pd=prob, risk_category=rcat,
                      model_version=ver, validation_status="ok"))
    db.add(__import__("backend.db", fromlist=["DecisionRecord"]).DecisionRecord(
        decision_id=str(uuid.uuid4()), application_id=app_id, recommendation=rec, policy_version=pver))
    db.commit(); db.close()
    return {"pd": prob, "risk_category": rcat, "recommendation": rec, "policy_version": pver,
            "model_version": ver, "application_id": app_id}

@router.post("/explain")
def explain(b: BorrowerInput):
    return shap_local(b.dict())
