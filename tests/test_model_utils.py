import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from winepredictor.model_utils import run_knn_grid_search


@pytest.fixture
def synthetic_regression_data():
    """Create a small synthetic dataset for KNN regression testing."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"f1": rng.rand(50), "f2": rng.rand(50), "f3": rng.rand(50)})
    y = pd.Series(X["f1"] * 3 + X["f2"] * 2 + rng.normal(0, 0.1, 50), name="target")
    return X, y


class TestRunKnnGridSearch:
    """Tests for run_knn_grid_search."""

    def test_returns_dict_with_expected_keys(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y, k_values=[1, 3, 5])
        expected_keys = {"best_k", "best_cv_score", "best_model", "best_train_score", "cv_results_df"}
        assert set(result.keys()) == expected_keys

    def test_best_k_is_int_in_range(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        k_vals = [2, 4, 6]
        result = run_knn_grid_search(X, y, k_values=k_vals)
        assert isinstance(result["best_k"], (int, np.integer))
        assert result["best_k"] in k_vals

    def test_scores_are_floats(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y, k_values=[3, 5])
        assert isinstance(result["best_cv_score"], float)
        assert isinstance(result["best_train_score"], float)

    def test_scores_in_reasonable_range(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y, k_values=[3, 5])
        assert -1.0 <= result["best_cv_score"] <= 1.0
        assert -1.0 <= result["best_train_score"] <= 1.0

    def test_best_model_is_pipeline(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y, k_values=[3])
        assert isinstance(result["best_model"], Pipeline)

    def test_cv_results_df_is_dataframe(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y, k_values=[1, 3, 5])
        assert isinstance(result["cv_results_df"], pd.DataFrame)
        assert len(result["cv_results_df"]) == 3

    def test_default_k_values(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        result = run_knn_grid_search(X, y)
        assert len(result["cv_results_df"]) == 29
        assert result["best_k"] in range(1, 30)

    def test_raises_on_empty_data(self):
        X = pd.DataFrame({"f1": [], "f2": []})
        y = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            run_knn_grid_search(X, y, k_values=[1, 3])
