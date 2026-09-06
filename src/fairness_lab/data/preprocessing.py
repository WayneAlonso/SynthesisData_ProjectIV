from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..settings import DatasetConfig


@dataclass
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_test: dict[str, pd.Series]
    sensitive_train: dict[str, pd.Series]
    transformer: ColumnTransformer
    feature_columns: list[str]


class DataPreprocessor:
    def __init__(self, dataset_cfg: DatasetConfig, test_size: float = 0.2, random_state: int = 42) -> None:
        self.dataset_cfg = dataset_cfg
        self.test_size = test_size
        self.random_state = random_state

    def prepare(self, df: pd.DataFrame) -> PreparedData:
        cleaned = df.copy()
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        train_df, test_df = train_test_split(
            cleaned,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=cleaned[self.dataset_cfg.target_column],
        )

        return self.prepare_from_split(train_df.reset_index(drop=True), test_df.reset_index(drop=True))

    def prepare_from_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> PreparedData:
        target_source = (self.dataset_cfg.target_transform or {}).get("source_column")
        excluded_columns = set(self.dataset_cfg.excluded_feature_columns or [])
        excluded_columns.add(self.dataset_cfg.target_column)
        if target_source:
            excluded_columns.add(target_source)
        available_numeric = [
            col for col in self.dataset_cfg.numerical_columns
            if col in train_df.columns and col not in excluded_columns and not train_df[col].isna().all()
        ]
        available_categorical = [
            col for col in self.dataset_cfg.categorical_columns
            if col in train_df.columns and col not in excluded_columns and not train_df[col].isna().all()
        ]
        feature_columns = available_numeric + available_categorical
        if not feature_columns:
            raise ValueError(
                f"No configured feature columns were found for dataset '{self.dataset_cfg.name}'. "
                "Please check configs/experiment.json against the raw file headers."
            )

        x_train_df = train_df[feature_columns].copy()
        x_test_df = test_df[feature_columns].copy()
        y_train = train_df[self.dataset_cfg.target_column].astype(int)
        y_test = test_df[self.dataset_cfg.target_column].astype(int)

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, available_numeric),
                ("cat", categorical_pipe, available_categorical),
            ]
        )

        X_train = transformer.fit_transform(x_train_df)
        X_test = transformer.transform(x_test_df)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        feature_names = transformer.get_feature_names_out()
        X_train = pd.DataFrame(X_train, columns=feature_names, index=train_df.index)
        X_test = pd.DataFrame(X_test, columns=feature_names, index=test_df.index)

        sensitive_test = {attr: test_df[attr].reset_index(drop=True) for attr in self.dataset_cfg.sensitive_attributes}
        sensitive_train = {attr: train_df[attr].reset_index(drop=True) for attr in self.dataset_cfg.sensitive_attributes}
        return PreparedData(
            train_df=train_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            sensitive_test=sensitive_test,
            sensitive_train=sensitive_train,
            transformer=transformer,
            feature_columns=feature_columns,
        )
