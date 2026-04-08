import pandas as pd

def extract_features_and_target(df, target_col):
    """
    Extract features (X) and target (y) from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features and target.
    target_col : str
        Name of the target column to separate.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.Series)
        A tuple (X, y) where X is the feature matrix and y is the target series.

    Raises
    ------
    KeyError
        If target_col is not found in the DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in the DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y
