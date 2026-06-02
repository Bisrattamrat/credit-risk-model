import pandas as pd


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


if __name__ == "__main__":

    df = load_data("data/raw/data.csv")

    print("Shape:", df.shape)

    print("\nMissing Values:")
    print(check_missing_values(df))

    print("\nSummary Statistics:")
    print(get_summary_statistics(df))