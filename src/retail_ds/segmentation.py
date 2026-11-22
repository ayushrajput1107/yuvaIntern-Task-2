"""Initiative I2: RFM customer segmentation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm(tx: pd.DataFrame) -> pd.DataFrame:
    snapshot = tx["order_date"].max() + pd.Timedelta(days=1)
    return (tx.groupby("customer_id")
              .agg(recency=("order_date", lambda s: (snapshot - s.max()).days),
                   frequency=("order_id", "nunique"),
                   monetary=("net_sales", "sum"))
              .reset_index())


def segment_customers(rfm: pd.DataFrame, n_clusters=5, seed=42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    out = rfm.copy()
    out["segment"] = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    return out


def segment_profile(seg: pd.DataFrame) -> pd.DataFrame:
    return (seg.groupby("segment")
               .agg(customers=("customer_id", "count"),
                    recency=("recency", "median"),
                    frequency=("frequency", "median"),
                    monetary=("monetary", "median"))
               .round(1))
