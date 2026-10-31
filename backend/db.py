from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from backend.config import settings
import datetime

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Applicant(Base):
    __tablename__ = "applicants"
    applicant_id = Column(String, primary_key=True)
    age = Column(Float)
    income = Column(Float)
    employment_years = Column(Float)
    credit_history_years = Column(Float)
    num_open_accounts = Column(Integer)
    num_times_late = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    applications = relationship("LoanApplication", back_populates="applicant")

class LoanApplication(Base):
    __tablename__ = "loan_applications"
    application_id = Column(String, primary_key=True)
    applicant_id = Column(String, ForeignKey("applicants.applicant_id"))
    loan_amount = Column(Float)
    loan_type = Column(String)
    loan_term = Column(Integer)
    interest_rate = Column(Float)
    status = Column(String, default="pending")
    application_ts = Column(DateTime, default=datetime.datetime.utcnow)
    applicant = relationship("Applicant", back_populates="applications")
    predictions = relationship("Prediction", back_populates="application")

class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id = Column(String, primary_key=True)
    application_id = Column(String, ForeignKey("loan_applications.application_id"))
    pd = Column(Float)
    risk_category = Column(String)
    model_version = Column(String)
    prediction_ts = Column(DateTime, default=datetime.datetime.utcnow)
    validation_status = Column(String, default="ok")
    explanation_ref = Column(String)
    application = relationship("LoanApplication", back_populates="predictions")

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_id = Column(String, primary_key=True)
    name = Column(String)
    version = Column(String)
    dataset_ref = Column(String)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
    metrics = Column(JSON)
    status = Column(String, default="registered")

class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"
    eval_id = Column(String, primary_key=True)
    model_version = Column(String)
    split_name = Column(String)
    metrics = Column(JSON)
    evaluated_at = Column(DateTime, default=datetime.datetime.utcnow)

class DecisionRecord(Base):
    __tablename__ = "decision_records"
    decision_id = Column(String, primary_key=True)
    application_id = Column(String, ForeignKey("loan_applications.application_id"))
    recommendation = Column(String)
    policy_version = Column(String)
    decision_ts = Column(DateTime, default=datetime.datetime.utcnow)
    review_status = Column(String, default="pending")

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    snapshot_id = Column(String, primary_key=True)
    period = Column(String)
    metrics = Column(JSON)
    model_version = Column(String)
    snapshot_ts = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
