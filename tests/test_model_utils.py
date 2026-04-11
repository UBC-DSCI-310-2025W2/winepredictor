import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from winepredictor.model_utils import run_knn_grid_search


@pytest.fixture
def synthetic_regression_data():
    """
    Create a small synthetic dataset for KNN regression testing.
    """
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"f1": rng.rand(50), "f2": rng.rand(50), "f3": rng.rand(50)})
    y = pd.Series(X["f1"] * 3 + X["f2"] * 2 + rng.normal(0, 0.1, 50), name="target")
    return X, y


def test_returns_dict_with_expected_keys(synthetic_regression_data):
    """
    Test that run_knn_grid_search returns a dictionary containing exactly
    the expected keys: best_k, best_cv_score, best_model, best_train_score,
    and cv_results_df. Ensures the output structure is consistent and complete.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y, k_values=[1, 3, 5])
    expected_keys = {"best_k", "best_cv_score", "best_model", "best_train_score", "cv_results_df"}
    assert set(result.keys()) == expected_keys

def test_best_k_is_int_in_range(synthetic_regression_data):
    """
    Test that the best_k value returned is an integer and is one of the
    k values that was passed in. Ensures the function selects a valid K
    from the provided search space rather than returning an out-of-range value.
    """
    X, y = synthetic_regression_data
    k_vals = [2, 4, 6]
    result = run_knn_grid_search(X, y, k_values=k_vals)
    assert isinstance(result["best_k"], (int, np.integer))
    assert result["best_k"] in k_vals

def test_scores_are_floats(synthetic_regression_data):
    """
    Test that both best_cv_score and best_train_score are returned as floats.
    Ensures scores are numeric and can be used in downstream comparisons
    or reporting without type conversion.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y, k_values=[3, 5])
    assert isinstance(result["best_cv_score"], float)
    assert isinstance(result["best_train_score"], float)

def test_scores_in_reasonable_range(synthetic_regression_data):
    """
    Test that both best_cv_score and best_train_score fall within the
    range [-1.0, 1.0], which is the valid range for R² scores. A score
    outside this range would indicate a severely broken model or data issue.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y, k_values=[3, 5])
    assert -1.0 <= result["best_cv_score"] <= 1.0
    assert -1.0 <= result["best_train_score"] <= 1.0

def test_best_model_is_pipeline(synthetic_regression_data):
    """
    Test that the best_model returned is a scikit-learn Pipeline object.
    The function is expected to wrap the KNN model in a Pipeline so it
    can be used directly for prediction without additional setup.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y, k_values=[3])
    assert isinstance(result["best_model"], Pipeline)

def test_cv_results_df_is_dataframe(synthetic_regression_data):
    """
    Test that cv_results_df is a pandas DataFrame with one row per K value
    tested. Ensures the full grid search results are returned in a structured
    format that can be used for analysis or plotting.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y, k_values=[1, 3, 5])
    assert isinstance(result["cv_results_df"], pd.DataFrame)
    assert len(result["cv_results_df"]) == 3

def test_default_k_values(synthetic_regression_data):
    """
    Test that when no k_values are provided, the function defaults to
    searching K from 1 to 29 (29 values total), and that the best_k
    falls within that default range.
    """
    X, y = synthetic_regression_data
    result = run_knn_grid_search(X, y)
    assert len(result["cv_results_df"]) == 29
    assert result["best_k"] in range(1, 30)

def test_raises_on_empty_data():
    """
    Test that a ValueError is raised when the input data is empty.
    An empty DataFrame and Series cannot be used to train a model,
    so the function should fail explicitly.
    """
    X = pd.DataFrame({"f1": [], "f2": []})
    y = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        run_knn_grid_search(X, y, k_values=[1, 3])