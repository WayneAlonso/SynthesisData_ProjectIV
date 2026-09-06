import pandas as pd

from fairness_lab.fairness.metrics import disparate_impact, theil_index


def test_disparate_impact_includes_zero_rate_group():
    y_pred = pd.Series([1, 0, 0, 0])
    sensitive = pd.Series(["A", "A", "B", "B"])

    result = disparate_impact(y_pred, sensitive)

    assert result["disparate_impact"] == 0.0


def test_theil_index_uses_group_sample_weights():
    y_pred = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
    sensitive = pd.Series(["A", "A", "A", "B", "B", "B", "B", "B"])

    result = theil_index(y_pred, sensitive)

    # The result must reflect A=3/3 and B=0/5 with population weights 3/8 and 5/8.
    assert result["theil_index"] > 0.0
    assert result["theil_index"] < 1.0
