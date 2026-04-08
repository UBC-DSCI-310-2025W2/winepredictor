import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from winepredictor.plot_utils import plot_correlation_heatmap, plot_quality_distribution


@pytest.fixture
def numeric_df():
    """Create a small synthetic dataset for heatmap correlation testing"""
    return pd.DataFrame(
        {
            "alcohol": [8.0, 9.0, 10.0, 11.0],
            "pH": [3.1, 3.2, 3.3, 3.4],
            "sulphates": [0.5, 0.6, 0.7, 0.8],
        }
    )


class TestRunPlotCorrelationHeatmap:
    """tests for plot_correlation_heatmap"""

    # test if output PNG file is created at the specified path
    def test_output_file_is_created_path(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        assert os.path.exists(output)

    # test if output file is a valid PNG
    def test_output_is_valid_png(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        with open(output, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG"

    # test if works with numeric only dataframe
    def test_works_with_numeric_only_dataframe(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)  # should not raise

    # test if it raises error with no numeric columns
    def test_raise_with_no_numeric_columns(self, tmp_path):
        df = pd.DataFrame({"type": ["red", "white"], "region": ["napa", "sonoma"]})
        output = str(tmp_path / "correlation_heatmap.png")
        with pytest.raises(ValueError, match="numeric"):
            plot_correlation_heatmap(df, output)

    # test custom figsize is respected
    def test_custom_figsize_is_respected(self, numeric_df, tmp_path):
        output = str(tmp_path / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output, figsize=(10, 9))
        img = plt.imread(output)
        assert img.shape[1] == 1000  # width: 10in * 100dpi
        assert img.shape[0] == 900  # height: 9in * 100dpi

    # creates parent directories if they do not exist
    def test_creates_parent_directories(self, numeric_df, tmp_path):
        output = str(tmp_path / "nested" / "dirs" / "correlation_heatmap.png")
        plot_correlation_heatmap(numeric_df, output)
        assert os.path.exists(output)


class TestPlotQualityDistribution:
    """Tests for plot_quality_distribution."""

    def test_output_file_is_created(self, tmp_path):
        df = pd.DataFrame({"quality": [3, 4, 4, 5, 6, 6, 7]})
        output_file = tmp_path / "quality_distribution.png"

        plot_quality_distribution(df, "quality", output_file)

        assert output_file.exists()
        assert output_file.is_file()

    def test_output_file_is_valid_png(self, tmp_path):
        df = pd.DataFrame({"quality": [3, 4, 5, 5, 6]})
        output_file = tmp_path / "quality_distribution.png"

        plot_quality_distribution(df, "quality", output_file)

        assert output_file.stat().st_size > 0
        with open(output_file, "rb") as f:
            magic_bytes = f.read(8)

        assert magic_bytes == b"\x89PNG\r\n\x1a\n"

    def test_raises_keyerror_if_column_missing(self, tmp_path):
        df = pd.DataFrame({"quality": [3, 4, 5]})
        output_file = tmp_path / "quality_distribution.png"

        with pytest.raises(KeyError):
            plot_quality_distribution(df, "missing_column", output_file)

    def test_works_with_different_column_names_and_dataframe_sizes(self, tmp_path):
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

    def test_creates_parent_directories_if_they_do_not_exist(self, tmp_path):
        df = pd.DataFrame({"quality": [3, 4, 5, 5, 6]})
        output_file = tmp_path / "nested" / "plots" / "quality_distribution.png"

        plot_quality_distribution(df, "quality", output_file)
