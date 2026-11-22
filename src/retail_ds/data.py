"""Synthetic retail data so every example runs without private company data."""
import numpy as np
import pandas as pd


def make_transactions(n_customers=2000, days=730, seed=42) -> pd.DataFrame:
    """One row per order: order_id, customer_id, order_date, net_sales."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    # customers differ in how often and how much they buy
    rate = rng.gamma(2.0, 0.01, n_customers)          # orders per day
    basket = rng.lognormal(3.4, 0.4, n_customers)     # typical spend
    rows = []
    for cid in range(n_customers):
        n = rng.poisson(rate[cid] * days)
        dates = start + pd.to_timedelta(rng.integers(0, days, n), unit="D")
        for d in dates:
            rows.append((cid, d, max(5.0, rng.normal(basket[cid], basket[cid] * 0.3))))
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "net_sales"])
    df = df.sort_values("order_date").reset_index(drop=True)
    df["order_id"] = np.arange(len(df))
    return df


def make_daily_sales(n_series=20, days=730, seed=42) -> pd.DataFrame:
    """Daily units by store-SKU with weekly/annual seasonality, price and promotions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    frames = []
    for s in range(n_series):
        level = rng.uniform(3, 20)
        price = rng.uniform(5, 30)
        promo = (rng.random(days) < 0.10).astype(int)
        dow = dates.dayofweek.values
        season = 1 + 0.25 * np.sin(2 * np.pi * dates.dayofyear.values / 365.25)
        weekend = 1 + 0.3 * (dow >= 5)
        mu = level * season * weekend * (1 + 0.5 * promo)
        units = rng.poisson(mu)
        frames.append(pd.DataFrame({"date": dates, "series_id": s, "units": units,
                                    "price": price * (1 - 0.2 * promo), "promo_flag": promo}))
    return pd.concat(frames, ignore_index=True)
