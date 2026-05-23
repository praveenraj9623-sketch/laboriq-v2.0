from __future__ import annotations

import pandas as pd


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean()*100), 2),
            "unique_values": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)
