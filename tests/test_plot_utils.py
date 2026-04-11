import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from winepredictor.plot_utils import plot_correlation_heatmap, plot_quality_distribution

@pytest.fixture
def numeric_df():
    """
    Create a small synthetic dataset for heatmap correlation testing
    """
    return pd.DataFrame(
        {
            "alcohol": [8.0, 9.0, 10.0, 11.0],
            "pH": [3.1, 3.2, 3.3, 3.4],
            "sulphates": [0.5, 0.6, 0.7, 0.8],
        }
    )


def test_output_file_is_created_path(numeric_df, tmp_path):
    """
    Test that calling plot_correlation_heatmap creates a PNG file at the
    exact output path provided. Verifies basic file creation behaviour.
    """
    output = str(tmp_path / "correlation_heatmap.png")
    plot_correlation_heatmap(numeric_df, output)
    assert os.path.exists(output)

def test_output_is_valid_png(numeric_df, tmp_path):
    """
    Test that the saved output file is a valid PNG by checking its magic
    bytes header (b'\\x89PNG').
    """
    output = str(tmp_path / "correlation_heatmap.png")
    plot_correlation_heatmap(numeric_df, output)
    with open(output, "rb") as f:
        magic = f.read(4)
    assert magic == b"\x89PNG"

def test_works_with_numeric_only_dataframe(numeric_df, tmp_path):
    """
    Test that the function runs without errors when given a DataFrame
    containing only numeric columns, which is the standard expected input.
    """
    output = str(tmp_path / "correlation_heatmap.png")
    plot_correlation_heatmap(numeric_df, output)  # should not raise

def test_raise_with_no_numeric_columns(tmp_path):
    """
    Test that a ValueError is raised when the input DataFrame contains no
    numeric columns. A correlation heatmap cannot be computed without
    numeric data, so the function should fail with a clear error.
    """
    df = pd.DataFrame({"type": ["red", "white"], "region": ["napa", "sonoma"]})
    output = str(tmp_path / "correlation_heatmap.png")
    with pytest.raises(ValueError, match="numeric"):
        plot_correlation_heatmap(df, output)

def test_custom_figsize_is_respected(numeric_df, tmp_path):
    """
    Test that passing a custom figsize parameter changes the dimensions of
    the saved PNG. Checks pixel dimensions of the output image to confirm
    the figsize is applied.
    """
    output = str(tmp_path / "correlation_heatmap.png")
    plot_correlation_heatmap(numeric_df, output, figsize=(10, 9))
    img = plt.imread(output)
    assert img.shape[1] == 1000  # width: 10in * 100dpi
    assert img.shape[0] == 900  # height: 9in * 100dpi

def test_creates_parent_directories(numeric_df, tmp_path):
    """
    Test that the function automatically creates any missing parent
    directories in the output path. This allows users to specify nested
    output paths without manually creating the directory structure first.
    """
    output = str(tmp_path / "nested" / "dirs" / "correlation_heatmap.png")
    plot_correlation_heatmap(numeric_df, output)
    assert os.path.exists(output)

    """Tests for plot_quality_distribution."""

def test_output_file_is_created(tmp_path):
    """
    Test that calling plot_quality_distribution creates a PNG file at the
    specified output path. Verifies basic file creation behaviour and that
    the output is a real file rather than a directory.
    """
    df = pd.DataFrame({"quality": [3, 4, 4, 5, 6, 6, 7]})
    output_file = tmp_path / "quality_distribution.png"

    plot_quality_distribution(df, "quality", output_file)

    assert output_file.exists()
    assert output_file.is_file()

def test_output_file_is_valid_png(tmp_path):
    """
    Test that the saved output file is a valid PNG by checking its full
    8-byte magic header and confirming the file size is greater than zero.
    Ensures the file is not empty or corrupted.
    """
    df = pd.DataFrame({"quality": [3, 4, 5, 5, 6]})
    output_file = tmp_path / "quality_distribution.png"

    plot_quality_distribution(df, "quality", output_file)

    assert output_file.stat().st_size > 0
    with open(output_file, "rb") as f:
        magic_bytes = f.read(8)

    assert magic_bytes == b"\x89PNG\r\n\x1a\n"

def test_raises_keyerror_if_column_missing(tmp_path):
    """
    Test that a KeyError is raised when the specified column does not exist
    in the DataFrame. Ensures the function fails explicitly rather than
    silently producing an empty or incorrect plot.
    """
    df = pd.DataFrame({"quality": [3, 4, 5]})
    output_file = tmp_path / "quality_distribution.png"

    with pytest.raises(KeyError):
        plot_quality_distribution(df, "missing_column", output_file)

def test_works_with_different_column_names_and_dataframe_sizes(tmp_path):
    """
    Test that the function works correctly with different column names and
    varying DataFrame sizes. Verifies the function is not hardcoded to a
    specific column name or dataset size.
    """
    small_df = pd.DataFrame({"rating": [1, 2, 2, 3]})
    large_df = pd.DataFrame({"score": list(range(100))})

    small_output = tmp_path / "small_plot.png"
    large_output = tmp_path / "large_plot.png"

    plot_quality_distribution(small_df, "rating", small_output)
    plot_quality_distribution(large_df, "score", large_output)

    assert small_output.exists()
    assert large_output.exists()
    assert small_output.stat().st_size > 0
    assert large_output.stat().st_size > 0

def test_creates_parent_directories_if_they_do_not_exist(tmp_path):
    """
    Test that the function automatically creates any missing parent directories
    in the output path. This allows users to specify nested output paths
    without manually creating the directory structure first.
    """
    df = pd.DataFrame({"quality": [3, 4, 5, 5, 6]})
    output_file = tmp_path / "nested" / "plots" / "quality_distribution.png"

    plot_quality_distribution(df, "quality", output_file)

    assert output_file.exists()
    assert output_file.parent.exists()