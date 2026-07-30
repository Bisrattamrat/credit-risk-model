import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_TYPES = ["int64", "float64"]
CATEGORICAL_TYPES = ["object"]


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset."""

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing values."""
    return df.isnull().sum()


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics."""
    return df.describe()


def create_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline."""

    numeric_features = df.select_dtypes(include=NUMERIC_TYPES).columns.tolist()

    categorical_features = df.select_dtypes(include=CATEGORICAL_TYPES).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


if __name__ == "main":
    df = load_data("data/raw/data.csv")
    print(df.head())