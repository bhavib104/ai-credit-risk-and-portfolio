import pandas as pd

def summary(df):
    return {
        "total_applications": int(len(df)),
        "total_exposure": float(df["AMT_CREDIT"].sum()) if "AMT_CREDIT" in df else 0.0,
        "high_risk_count": int((df.get("risk_category") == "high").sum()) if "risk_category" in df else 0,
        "risk_distribution": df["risk_category"].value_counts().to_dict() if "risk_category" in df else {},
        "by_loan_type": df.groupby("NAME_CONTRACT_TYPE")["AMT_CREDIT"].sum().to_dict() if "NAME_CONTRACT_TYPE" in df else {},
        "by_income_group": pd.cut(df["AMT_INCOME_TOTAL"], bins=5).value_counts().to_dict() if "AMT_INCOME_TOTAL" in df else {},
    }
