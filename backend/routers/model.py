from fastapi import APIRouter
from backend.db import SessionLocal, ModelRegistry, ModelEvaluation

router = APIRouter()

@router.get("/model/info")
def info():
    db = SessionLocal()
    m = db.query(ModelRegistry).order_by(ModelRegistry.trained_at.desc()).first()
    db.close()
    if not m:
        return {"status": "no model"}
    return {"name": m.name, "version": m.version, "status": m.status, "metrics": m.metrics, "trained_at": str(m.trained_at)}

@router.get("/model/metrics")
def metrics():
    db = SessionLocal()
    ev = db.query(ModelEvaluation).order_by(ModelEvaluation.evaluated_at.desc()).first()
    db.close()
    return {"evaluation": ev.metrics if ev else None}
