from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_levels: list[list[str]] | None = None,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean"))]
    )
    onehot_kwargs = {
        "handle_unknown": "ignore",
        "drop": "first",
    }
    if categorical_levels is not None:
        onehot_kwargs["categories"] = categorical_levels
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(**onehot_kwargs)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def _resolve_categorical_levels(
    category_map: dict[str, list[str]] | None,
    categorical_features: list[str],
) -> list[list[str]] | None:
    if not category_map or not categorical_features:
        return None
    ordered_levels: list[list[str]] = []
    for feature in categorical_features:
        levels = category_map.get(feature)
        if not isinstance(levels, list) or not levels:
            return None
        ordered_levels.append([str(level) for level in levels])
    return ordered_levels


def _filter_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    max_missing_rate: float,
) -> tuple[list[str], list[str], dict, list[str]]:
    features = numeric_features + categorical_features
    missing_rates = df[features].isna().mean().to_dict()
    drop = [f for f, rate in missing_rates.items() if rate > max_missing_rate]
    kept_numeric = [f for f in numeric_features if f not in drop]
    kept_categorical = [f for f in categorical_features if f not in drop]
    return kept_numeric, kept_categorical, missing_rates, drop
