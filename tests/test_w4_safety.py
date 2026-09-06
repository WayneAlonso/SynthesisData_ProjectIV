import numpy as np
import pandas as pd

from w4_synth import W4Synth


def test_privacy_proxy_does_not_change_target_or_categories():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "category": [0, 1, 0, 1], "value": [1.0, 2.0, 3.0, 4.0]})
    result = W4Synth()._apply_dp_noise(
        df, 0.3, np.random.default_rng(42), protected_columns={"target", "category"}
    )

    assert result["target"].tolist() == [0, 1, 0, 1]
    assert result["category"].tolist() == [0, 1, 0, 1]
    assert not result["value"].equals(df["value"])
