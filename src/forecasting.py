from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def _fit_scipy_optimized_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a small regularized forecasting model using SciPy optimization.

    This is intentionally transparent for interviews: we minimize mean squared
    error plus a small L2 penalty so coefficients do not become unstable on
    small portfolio datasets.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = np.zeros(X.shape[1], dtype=float)
    if X.shape[0] >= X.shape[1]:
        try:
            x0 = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            x0 = np.zeros(X.shape[1], dtype=float)

    def objective(beta: np.ndarray) -> float:
        preds = X @ beta
        mse = np.mean((y - preds) ** 2)
        penalty = 0.01 * np.sum(beta[1:] ** 2)
        return float(mse + penalty)

    result = minimize(objective, x0=x0, method="BFGS")
    beta = result.x if result.success else x0
    return beta, float(objective(beta))


def build_monthly_skill_panel(monthly_skill_demand: pd.DataFrame) -> pd.DataFrame:
    if monthly_skill_demand.empty:
        return pd.DataFrame(columns=["month", "skill", "mentions"])
    df = monthly_skill_demand.copy()
    df["month"] = pd.to_datetime(df["month"])
    skills = sorted(df["skill"].dropna().unique())
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    idx = pd.MultiIndex.from_product([months, skills], names=["month", "skill"])
    panel = df.groupby(["month", "skill"], as_index=True)["mentions"].sum().reindex(idx, fill_value=0).reset_index()
    panel["month_index"] = panel.groupby("skill").cumcount()
    panel["rolling_3m"] = panel.groupby("skill")["mentions"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    panel["rolling_6m"] = panel.groupby("skill")["mentions"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    panel["pct_change_3m"] = panel.groupby("skill")["rolling_3m"].pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0)
    return panel


def detect_emerging_skills(panel: pd.DataFrame, min_total_mentions: int = 5) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows=[]
    max_month = panel["month"].max()
    for skill, g in panel.groupby("skill"):
        g = g.sort_values("month")
        total = int(g["mentions"].sum())
        if total < min_total_mentions:
            continue
        early = g.head(max(1, len(g)//3))["mentions"].mean()
        recent = g.tail(max(3, len(g)//4))["mentions"].mean()
        x = g["month_index"].values.reshape(-1,1)
        y = g["mentions"].values
        slope = float(LinearRegression().fit(x,y).coef_[0]) if len(g) >= 2 else 0.0
        growth_score = (recent + 1) / (early + 1)
        rows.append({
            "skill": skill,
            "total_mentions": total,
            "early_avg_mentions": round(float(early), 3),
            "recent_avg_mentions": round(float(recent), 3),
            "linear_slope": round(slope, 4),
            "growth_score": round(float(growth_score), 4),
            "trend_label": "Emerging" if growth_score >= 1.5 and slope > 0 else ("Declining" if growth_score <= 0.75 and slope < 0 else "Stable"),
            "last_observed_month": max_month.strftime("%Y-%m"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["trend_label", "growth_score", "linear_slope"], ascending=[True, False, False])


def forecast_skill_demand(panel: pd.DataFrame, periods: int = 6, top_n: int = 20) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    totals = panel.groupby("skill")["mentions"].sum().sort_values(ascending=False).head(top_n)
    forecast_rows=[]
    for skill in totals.index:
        g = panel[panel["skill"] == skill].sort_values("month").copy()
        if len(g) < 3:
            continue
        g["lag_1"] = g["mentions"].shift(1).fillna(0)
        g["lag_2"] = g["mentions"].shift(2).fillna(0)
        train = g.copy()
        X = train[["month_index", "lag_1", "lag_2", "rolling_3m"]].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        y = train["mentions"].to_numpy(dtype=float)
        beta, objective_value = _fit_scipy_optimized_regression(X, y)
        fitted = X @ beta
        mae = mean_absolute_error(y, fitted)
        last_month = g["month"].max()
        history = list(g["mentions"].tail(2))
        rolling = float(g["rolling_3m"].iloc[-1])
        for step in range(1, periods + 1):
            next_month = last_month + pd.DateOffset(months=step)
            next_index = int(g["month_index"].max()) + step
            lag_1 = history[-1] if history else 0
            lag_2 = history[-2] if len(history) > 1 else 0
            x_next = np.array([1.0, float(next_index), float(lag_1), float(lag_2), float(rolling)])
            pred = float(x_next @ beta)
            pred = max(0.0, pred)
            forecast_rows.append({
                "skill": skill,
                "forecast_month": next_month.strftime("%Y-%m"),
                "forecast_mentions": round(pred, 2),
                "model_mae": round(float(mae), 3),
                "optimization_objective": round(float(objective_value), 4),
                "forecast_method": "scipy_optimized_regression_with_lags",
                "scipy_function": "scipy.optimize.minimize",
            })
            history.append(pred)
            rolling = float(np.mean(history[-3:]))
    return pd.DataFrame(forecast_rows)


def build_workforce_forecast(monthly_skill_demand: pd.DataFrame, periods: int = 6) -> dict[str, pd.DataFrame]:
    panel = build_monthly_skill_panel(monthly_skill_demand)
    trends = detect_emerging_skills(panel)
    forecasts = forecast_skill_demand(panel, periods=periods)
    return {"skill_monthly_panel": panel, "skill_trends": trends, "skill_forecasts": forecasts}
