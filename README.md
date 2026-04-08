# winepredictor

[![Tests](https://github.com/UBC-DSCI-310-2025W2/winepredictor/actions/workflows/tests.yml/badge.svg)](https://github.com/UBC-DSCI-310-2025W2/winepredictor/actions/workflows/tests.yml)

A Python package with utility functions for wine quality prediction — data preparation, KNN model tuning, and visualization.

## Overview

`winepredictor` provides reusable functions for a wine quality prediction workflow:

- **`extract_features_and_target(df, target_col)`** — Split a DataFrame into feature matrix X and target series y.
- **`run_knn_grid_search(X_train, y_train, k_values=None)`** — Run KNN regression with grid search cross-validation over K values.
- **`plot_correlation_heatmap(df, output_path, figsize=(8,7))`** — Generate and save a correlation heatmap as PNG.
- **`plot_quality_distribution(df, col, output_path)`** — Plot and save the distribution of a column as a bar chart PNG.

## Where This Fits in the Ecosystem

Several Python packages provide overlapping functionality:

| Package | Scope | Difference |
|---|---|---|
| **scikit-learn** | General ML library | `winepredictor` wraps a specific KNN grid search pipeline with sensible defaults and structured output |
| **matplotlib / seaborn** | General plotting | `winepredictor` provides opinionated, single-call plotting functions that save to disk |
| **pandas** | Data manipulation | `winepredictor.extract_features_and_target` is a convenience wrapper for the common split-and-drop pattern |

`winepredictor` is not a replacement for any of these — it is a thin, domain-specific layer that bundles common operations for wine quality analysis into a single importable package.

## Installation

```bash
pip install git+https://github.com/UBC-DSCI-310-2025W2/winepredictor.git@v0.1.0
```

## Usage

```python
import pandas as pd
from winepredictor import extract_features_and_target, run_knn_grid_search
from winepredictor import plot_correlation_heatmap, plot_quality_distribution

# Load data
df = pd.read_csv("data/train.csv")

# Split features and target
X_train, y_train = extract_features_and_target(df, "quality")

# Run KNN grid search
results = run_knn_grid_search(X_train, y_train, k_values=list(range(1, 30)))
print(f"Best K: {results['best_k']}, CV R²: {results['best_cv_score']:.3f}")

# Generate plots
plot_correlation_heatmap(df, "results/correlation_heatmap.png")
plot_quality_distribution(df, "quality", "results/quality_distribution.png")
```

## Running Tests

```bash
pip install ".[dev]"
pytest tests/ -v
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
