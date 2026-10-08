import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path.cwd().parent
sys.path.append(str(ROOT))



df = pd.read_csv(RAW_DATA_PATH)


def process_df(df):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    cat_features = [col for col in df.columns if df[col].dtype == "str"]
    num_features = [col for col in df.columns if df[col].dtype != "str"]

    age_to_drop = df[df["person_age"] > 100].index
    df = df.drop(age_to_drop).copy()

    ordinal_features = ["loan_grade"]
    nominal_features = ["loan_intent", "person_home_ownership"]
    binary_features = ["cb_person_default_on_file"]

    transformer = ColumnTransformer(
        transformers=[
        ("median_imputer", SimpleImputer(strategy="median"), ["person_emp_length"]),
        ("mean_imputer", SimpleImputer(strategy="mean"), ["loan_int_rate"]),
        ("onehot", OneHotEncoder(drop="first"), nominal_features),
        ("binary", OrdinalEncoder(categories=[["N", "Y"]]), binary_features),
        ("ordinal", OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F", "G"]]), ordinal_features)],
        remainder="passthrough"
    )

    X_train_preprocessed = transformer.fit_transform(X_train)
    column_names = transformer.get_feature_names_out()
    X_test_preprocessed = transformer.transform(X_test)
    X_train_preprocessed = pd.DataFrame(data=X_train_preprocessed, columns=column_names)
    X_test_preprocessed = pd.DataFrame(data=X_test_preprocessed, columns=column_names)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(PROCESSED_DATA_DIR / "X_train_preprocessed.parquet")
    X_test.to_parquet(PROCESSED_DATA_DIR / "X_test_preprocessed.parquet")

    y_train.to_frame().to_parquet(
        PROCESSED_DATA_DIR / "y_train.parquet"
    )

    y_test.to_frame().to_parquet(
        PROCESSED_DATA_DIR / "y_test.parquet"
    )





if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_df(df)
    save_processed_data(X_train, X_test, y_train, y_test)