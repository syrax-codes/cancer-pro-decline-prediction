"""Loading data and building the preprocessing step.

The split and preprocessing reproduce notebooks/02_preprocessing.ipynb exactly
(stratified 80/20, random_state=42, median imputation + missing indicators +
scaling, one-hot categories). We rebuild it inside each model pipeline instead
of loading models/preprocessor.joblib so that:
  * the saved model takes raw patient values (what the Streamlit app collects);
  * cross-validation re-fits imputation/scaling inside every fold (no leakage);
  * it works on any scikit-learn version (the pickled preprocessor does not).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA)


def get_xy(df: pd.DataFrame, target: str = config.DEFAULT_TARGET, features=None):
    features = features or config.FEATURE_COLS
    X = df[features].copy()
    y = df[target].astype(int)
    return X, y


def split(target: str = config.DEFAULT_TARGET):
    """Same split as Person 1's processed files (verified in tests)."""
    df = load_raw()
    X = df.drop(columns=config.LEAKAGE_COLS)[config.FEATURE_COLS]
    y = df[target].astype(int)
    # Stratify on drop_outcome like the preprocessing notebook so the patients in
    # train/test are identical whichever target we predict.
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df["drop_outcome"],
    )


def build_preprocessor(numeric_cols=None, categorical_cols=None) -> ColumnTransformer:
    numeric_cols = config.NUMERIC_COLS if numeric_cols is None else numeric_cols
    categorical_cols = config.CATEGORICAL_COLS if categorical_cols is None else categorical_cols
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                             list(categorical_cols)))
    return ColumnTransformer(transformers, remainder="drop")
