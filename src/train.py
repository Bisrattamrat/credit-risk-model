import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow


df = pd.read_csv(
    "data/processed/processed_data.csv"
)

X = df.select_dtypes(
    include=["int64", "float64"]
).drop(
    columns=["is_high_risk"],
    errors="ignore"
)

y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "LogisticRegression":
        LogisticRegression(max_iter=1000),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

best_model = None
best_auc = 0

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            auc
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model

joblib.dump(
    best_model,
    "best_model.pkl"
)

print(
    "Training completed."
)