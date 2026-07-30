import pandas as pd

from src.data_processing import (
    check_missing_values,
    create_pipeline,
    get_summary_statistics,
    load_data,
)


def test_missing_values():
    df = pd.DataFrame({"A":[1,None,3]})
    result = check_missing_values(df)
    assert result["A"] == 1


def test_summary_statistics():
    df = pd.DataFrame({"Amount":[10,20,30]})
    result = get_summary_statistics(df)
    assert "Amount" in result.columns


def test_pipeline_created():
    df = pd.DataFrame({
        "Amount":[1,2],
        "Country":["UG","KE"]
    })
    pipeline = create_pipeline(df)
    assert pipeline is not None


def test_load_data():
    df = load_data("data/raw/data.csv")
    assert len(df) > 0


def test_dataframe_type():
    df = load_data("data/raw/data.csv")
    assert isinstance(df, pd.DataFrame)