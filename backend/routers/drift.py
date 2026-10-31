from fastapi import APIRouter
from monitoring.drift import prediction_drift
import numpy as np

router = APIRouter()

@router.get("/drift/check")
def drift_check():
    # demo: compare two synthetic samples
    a = np.random.default_rng(1).random(200)
    b = np.random.default_rng(2).random(200)
    return prediction_drift(a, b)
