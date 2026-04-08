from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_correlation_heatmap(df, output_path, figsize=(8, 7)):
    """
    Plot a correlation heatmap and save it as a PNG.

    Parameters:
    ---
        df: pd.DataFrame
            The input DataFrame containing the data to analyze with numeric values
        output_path: str
            File path where the PNG will be saved
        figsize: tuple, optional
            Figure size as (width, height)
    Returns:
    ---
    None (saves PNG file)

    Example:
    ---
    >>> df = pd.read_csv('wine_predictions')
    >>> filepath = "../results/figures"
    >>> plot_correlation_heatmap(df, filepath)


    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:  # check if dataframe contains at least one numeric column
        raise ValueError("DataFrame must contain at least one numeric column.")

    # creates parent directory for output file if it does not already exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)  # sets fig size
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Heatmap of Wine Features")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_quality_distribution(df, col, output_path):
    """
    Plot the distribution of values in a DataFrame column and save as a PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    col : str
        Column name to plot.
    output_path : str
        File path where the PNG will be saved.

    Returns
    -------
    None
        Saves the plot to disk.

    Raises
    ------
    KeyError
        If the column does not exist in the dataframe.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    value_counts = df[col].value_counts().sort_index()

    plt.figure()
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Distribution of {col.capitalize()}")
    plt.xlabel(col.capitalize())
    plt.ylabel("Count")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
