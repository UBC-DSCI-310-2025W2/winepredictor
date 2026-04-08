import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline


def run_knn_grid_search(X_train, y_train, k_values=None):
    """Run KNN grid search with cross-validation and return results.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    k_values : list of int, optional
        Values of K to search over. Defaults to range(1, 30).

    Returns
    -------
    dict
        Keys: best_k, best_cv_score, best_model, best_train_score, cv_results_df
    """
    if k_values is None:
        k_values = list(range(1, 30))

    pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
    param_grid = {"kneighborsregressor__n_neighbors": k_values}
    grid = GridSearchCV(pipe, param_grid, n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)

    best_k = grid.best_params_["kneighborsregressor__n_neighbors"]
    best_cv_score = float(grid.best_score_)
    best_model = grid.best_estimator_
    best_train_score = float(best_model.score(X_train, y_train))

    cv_results_df = pd.DataFrame(grid.cv_results_)[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
        ]
    ]

    return {
        "best_k": best_k,
        "best_cv_score": best_cv_score,
        "best_model": best_model,
        "best_train_score": best_train_score,
        "cv_results_df": cv_results_df,
    }
