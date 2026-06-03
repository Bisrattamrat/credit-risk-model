import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/raw/data.csv")

df["TransactionStartTime"] = pd.to_datetime(
    df["TransactionStartTime"]
)

snapshot_date = (
    df["TransactionStartTime"].max()
    + pd.Timedelta(days=1)
)

rfm = (
    df.groupby("CustomerId")
    .agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date - x.max()
            ).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Amount",
            "sum"
        )
    )
    .reset_index()
)

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ]
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

risk_cluster = (
    rfm.groupby("Cluster")[
        ["Frequency", "Monetary"]
    ]
    .mean()
    .sum(axis=1)
    .idxmin()
)

rfm["is_high_risk"] = (
    rfm["Cluster"] == risk_cluster
).astype(int)

df = df.merge(
    rfm[
        [
            "CustomerId",
            "is_high_risk"
        ]
    ],
    on="CustomerId",
    how="left"
)

df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("processed_data.csv created")