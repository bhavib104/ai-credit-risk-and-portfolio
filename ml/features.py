import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SENSITIVE = ["CODE_GENDER"]

NUMERIC = ["AMT_INCOME_TOTAL","AMT_CREDIT","age","employment_years","credit_history_years",
           "num_open_accounts","num_times_late","dependents","loan_term","interest_rate",
           "credit_income_ratio","annuity_proxy"]
CATEGORICAL = ["NAME_CONTRACT_TYPE","FLAG_OWN_CAR","FLAG_OWN_REALTY"]

def engineer(df):
    d = df.copy()
    d["age"] = (-d["DAYS_BIRTH"] / 365.25).clip(18, 100)
    d["employment_years"] = (-d["DAYS_EMPLOYED"] / 365.25).clip(0, 50).replace([np.inf], 0)
    d["dependents"] = d["CNT_CHILDREN"]
    d["credit_income_ratio"] = d["AMT_CREDIT"] / (d["AMT_INCOME_TOTAL"] + 1)
    d["annuity_proxy"] = d["AMT_CREDIT"] / (d["loan_term"] + 1)
    return d

def build_preprocessor():
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, NUMERIC), ("cat", cat_pipe, CATEGORICAL)])
    return pre

def feature_columns():
    return NUMERIC + CATEGORICAL
