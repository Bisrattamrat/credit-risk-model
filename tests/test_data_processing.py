import pandas as pd

from src.data_processing import (
    check_missing_values,
    get_summary_statistics
)


def test_missing_values_function():

    df = pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": [4, 5, 6]
        }
    )

    result = check_missing_values(df)

    assert result["A"] == 1


def test_summary_statistics():

    df = pd.DataFrame(
        {
            "Amount": [10, 20, 30]
        }
    )

    result = get_summary_statistics(df)

    assert "Amount" in result.columns