from research_main import partition_datasets
from fairness_lab.settings import Settings


def test_holdout_is_excluded_from_training_modules():
    training, holdouts = partition_datasets(
        ["sbo", "sbo_withheld", "acs_ma_2019"],
        Settings(),
    )

    assert training == ["sbo", "acs_ma_2019"]
    assert holdouts == ["sbo_withheld"]
