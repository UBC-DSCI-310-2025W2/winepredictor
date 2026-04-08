import pytest
import pandas as pd

@pytest.fixture
def sample_wine_data():
    """Returns a small sample of the wine quality dataset for testing."""
    data = {
        "fixed_acidity": [7.4, 7.8],
        "volatile_acidity": [0.70, 0.88],
        "citric_acid": [0.00, 0.00],
        "residual_sugar": [1.9, 2.6],
        "chlorides": [0.076, 0.098],
        "free_sulfur_dioxide": [11.0, 25.0],
        "total_sulfur_dioxide": [34.0, 67.0],
        "density": [0.9978, 0.9968],
        "pH": [3.51, 3.20],
        "sulphates": [0.56, 0.68],
        "alcohol": [9.4, 9.8],
        "quality": [5, 5]
    }
    return pd.DataFrame(data)
