"""Synthetic credit dataset generator (MVP). Mimics Home Credit-style fields."""
import numpy as np
import pandas as pd
from backend.config import settings

RANDOM_SEED = 42

def generate(n=20000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = rng.lognormal(10.5, 0.6, n).clip(10000, 300000)
    employment_years = rng.exponential(5, n).clip(0, 40)
    credit_history_years = rng.exponential(6, n).clip(0, 40)
    num_open_accounts = rng.poisson(4, n).clip(0, 20)
    num_times_late = rng.poisson(0.5, n).clip(0, 10)
    loan_amount = rng.lognormal(10.0, 0.7, n).clip(1000, 100000)
    loan_type = rng.choice(["cash_loan", "revolving_loan"], n, p=[0.8, 0.2])
    loan_term = rng.choice([12, 24, 36, 48, 60], n)
    interest_rate = rng.uniform(5, 25, n)
    dependents = rng.poisson(1, n).clip(0, 6)
    has_car = rng.choice([0, 1], n, p=[0.6, 0.4])
    has_realty = rng.choice([0, 1], n, p=[0.7, 0.3])
    gender = rng.choice(["M", "F"], n)

    # default probability driven by risk factors
    logit = (
        -3.0
        + 0.02 * (loan_amount / (income + 1) * 10)
        + 0.3 * num_times_late
        - 0.05 * employment_years
        - 0.03 * credit_history_years
        + 0.00001 * (income < 30000) * 1000
        + 0.4 * (num_open_accounts > 8)
        + 0.00002 * loan_amount
    )
    p = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "TARGET": target,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": loan_amount,
        "NAME_CONTRACT_TYPE": loan_type,
        "DAYS_EMPLOYED": -(employment_years * 365).astype(int),
        "DAYS_BIRTH": -(age * 365).astype(int),
        "CNT_CHILDREN": dependents,
        "FLAG_OWN_CAR": has_car,
        "FLAG_OWN_REALTY": has_realty,
        "CODE_GENDER": gender,
        "num_open_accounts": num_open_accounts,
        "num_times_late": num_times_late,
        "credit_history_years": credit_history_years,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
    })
    return df

def load_or_generate(path=None):
    import os
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return generate()
