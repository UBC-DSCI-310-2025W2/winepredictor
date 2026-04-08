import pytest
import pandas as pd
from winepredictor.data_utils import extract_features_and_target

def test_extract_features_and_target_returns_tuple(sample_wine_data):
    result = extract_features_and_target(sample_wine_data, "quality")
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_extract_features_and_target_correct_shape(sample_wine_data):
    X, y = extract_features_and_target(sample_wine_data, "quality")
    assert X.shape[1] == sample_wine_data.shape[1] - 1
    assert "quality" not in X.columns

def test_extract_features_and_target_values(sample_wine_data):
    X, y = extract_features_and_target(sample_wine_data, "quality")
    pd.testing.assert_series_equal(y, sample_wine_data["quality"])

def test_extract_features_and_target_raises_error(sample_wine_data):
    with pytest.raises(KeyError):
        extract_features_and_target(sample_wine_data, "wrong_column")
