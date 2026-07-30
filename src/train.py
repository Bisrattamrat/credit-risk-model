import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "best_model.pkl"
RANDOM_STATE = 42


def train_model() -> None:
    """Train and save the best performing model."""

    df = pd.read_csv(DATA_PATH)

    X = df.select_dtypes(include=["int64", "float64"]).drop(
        columns=["is_high_risk"],
        errors="ignore",
    )

    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    best_model = None
    best_auc = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        print(f"\n{name}")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(f"ROC AUC  : {auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    joblib.dump(best_model, MODEL_PATH)

    print("\nTraining completed.")
    print(f"Best ROC AUC: {best_auc:.3f}")


if __name__ == "__main__":
    train_model()