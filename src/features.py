"""Feature selection and model-ready transformations."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Build leakage-safe categorical and numeric transformations."""
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipeline, categorical),
        ("numeric", numeric_pipeline, numeric),
    ])
