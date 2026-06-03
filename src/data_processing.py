import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Load transaction data from CSV.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )


def check_missing_values(df):
    """
    Return missing value counts.
    """
    return df.isnull().sum()


def get_summary_statistics(df):
    """
    Return summary statistics.
    """
    return df.describe()


def create_pipeline(df):
    """
    Create preprocessing pipeline.
    """

    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


if __name__ == "__main__":

    df = load_data("data/raw/data.csv")

    print("Shape:", df.shape)

    print("\nMissing Values:")
    print(check_missing_values(df))

    print("\nSummary Statistics:")
    print(get_summary_statistics(df))

    pipeline = create_pipeline(df)

    print("\nPipeline Created Successfully")