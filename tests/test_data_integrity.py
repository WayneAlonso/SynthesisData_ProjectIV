import pandas as pd

from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.settings import DatasetConfig


def test_target_source_is_excluded_from_model_features():
    cfg = DatasetConfig(
        name="demo",
        path="demo.csv",
        target_column="target",
        target_transform={"type": "threshold", "source_column": "income", "threshold": 10},
        sensitive_attributes=["group"],
        categorical_columns=["group"],
        numerical_columns=["income", "other"],
    )
    df = pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "income": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
            "other": range(10),
            "group": ["A", "B"] * 5,
        }
    )

    prepared = DataPreprocessor(cfg, test_size=0.2, random_state=42).prepare(df)

    assert "income" not in prepared.feature_columns
    assert "other" in prepared.feature_columns
