"""Initiative I1: demand forecasting with rolling-origin validation."""
import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["lag_7", "lag_28", "roll_mean_28", "price", "promo_flag", "dow", "week_of_year"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["series_id", "date"]).copy()
    g = df.groupby("series_id")["units"]
    df["lag_7"] = g.shift(7)
    df["lag_28"] = g.shift(28)
    df["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["dow"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df.dropna(subset=["lag_28", "roll_mean_28"])


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual, pred) -> float:
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float((pred.sum() - actual.sum()) / actual.sum())


def rolling_origin_eval(df: pd.DataFrame, cutoffs, horizon_days=28) -> pd.DataFrame:
    """Train before each cutoff, test on the next `horizon_days`. Compare with a seasonal-naive baseline."""
    rows = []
    for cutoff in pd.to_datetime(cutoffs):
        train = df[df.date < cutoff]
        valid = df[(df.date >= cutoff) & (df.date < cutoff + pd.Timedelta(days=horizon_days))]
        model = lgb.LGBMRegressor(objective="tweedie", n_estimators=300, learning_rate=0.05,
                                  random_state=42, verbose=-1)
        model.fit(train[FEATURES], train["units"])
        pred = model.predict(valid[FEATURES])
        naive = valid["lag_7"].values        # seasonal naive: same weekday last week
        rows.append({"cutoff": cutoff.date(),
                     "wape_model": wape(valid["units"], pred),
                     "wape_baseline": wape(valid["units"], naive),
                     "bias_model": bias(valid["units"], pred)})
    out = pd.DataFrame(rows)
    out["improvement_vs_baseline"] = 1 - out["wape_model"] / out["wape_baseline"]
    return out.round(4)
