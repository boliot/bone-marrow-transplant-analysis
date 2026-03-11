import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
def load_data(filepath: str) -> pd.DataFrame:
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ["float32", "float64", "int32", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns
    numeric_cols = [c for c in numeric_cols if c != "survival_status"]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    target = "survival_status"
    feature_cols = [c for c in df.columns if c != target]

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropping highly correlated features (threshold={threshold}): {to_drop}")
    df = df.drop(columns=to_drop)
    return df
def split_and_balance(df: pd.DataFrame, target: str = "survival_status",
                      test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]

    # Convert target to int if needed
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"Before SMOTE — Train class distribution:\n{y_train.value_counts()}")
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE — Train class distribution:\n{pd.Series(y_train).value_counts()}")

    return X_train, X_test, y_train, y_test

def preprocess_pipeline(filepath: str):
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_features(df)
    df = drop_correlated_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/bone-marrow.arff"
    X_train, X_test, y_train, y_test = preprocess_pipeline(path)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")