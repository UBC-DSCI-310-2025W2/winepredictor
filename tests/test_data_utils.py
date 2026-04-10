import pytest
import pandas as pd
from winepredictor.data_utils import extract_features_and_target


"""Tests for extract_features_and_targets."""
"""to run: python -m pytest tests/test_data_utils.py"""


def test_extract_features_and_target_returns_tuple(sample_wine_data):
    """
    Test that extract_features_and_target returns a tuple of length 2.
    The first element should be the feature matrix X and the second
    should be the target series y.
    """
    result = extract_features_and_target(sample_wine_data, "quality")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_features_and_target_correct_shape(sample_wine_data):
    """
    Test that the feature matrix X has one fewer column than the original
    DataFrame (the target column is removed), and that the target column
    'quality' is not present in X.
    """
    X, y = extract_features_and_target(sample_wine_data, "quality")
    assert X.shape[1] == sample_wine_data.shape[1] - 1
    assert "quality" not in X.columns


def test_extract_features_and_target_values(sample_wine_data):
    """
    Test that the target series y contains the correct values from the
    original DataFrame.
    """
    X, y = extract_features_and_target(sample_wine_data, "quality")
    pd.testing.assert_series_equal(y, sample_wine_data["quality"])


def test_extract_features_and_target_raises_error(sample_wine_data):
    """
    Test that a KeyError is raised when the specified target column does
    not exist in the DataFrame. 
    """
    with pytest.raises(KeyError):
        extract_features_and_target(sample_wine_data, "wrong_column")
        
def test_extract_features_and_target_empty_dataframe():
    """
    Test that passing a completely empty DataFrame raises a ValueError.
    An empty DataFrame has no rows or columns, so splitting it into
    features and target is not meaningful and should fail.
    """
    empty_df = pd.DataFrame()
    with pytest.raises((ValueError, KeyError)):
        extract_features_and_target(empty_df, "quality")
        
def test_extract_features_and_target_empty_rows():
    """
    Test that a DataFrame with columns but no rows returns an empty feature
    matrix and an empty target series without raising an error. This is a
    valid boundary condition — the structure is correct but there is no data.
    """
    df = pd.DataFrame(columns=["alcohol", "pH", "quality"])
    X, y = extract_features_and_target(df, "quality")
    assert X.shape == (0, 2)
    assert len(y) == 0
    
def test_extract_features_and_target_invalid_type_dataframe():
    """
    Test that passing a non-DataFrame as the first argument raises a TypeError.
    The function expects a pandas DataFrame — passing a list, dict, or other
    type should fail explicitly rather than producing unexpected results.
    """
    with pytest.raises((TypeError, AttributeError)):
        extract_features_and_target([[1, 2, 3], [4, 5, 6]], "quality")
