import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, OrdinalEncoder
import pickle
from sklearn.model_selection import TunedThresholdClassifierCV
import sys
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, RAW_DATA_PATH, API_PATH
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

X_train_preprocessed = pd.read_parquet(PROCESSED_DATA_DIR / "X_train_preprocessed.parquet")
X_test_preprocessed = pd.read_parquet(PROCESSED_DATA_DIR / "X_test_preprocessed.parquet")
y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")


classifier_tuned = TunedThresholdClassifierCV(estimator=XGBClassifier(), scoring="balanced_accuracy", thresholds=100, cv=5, n_jobs=-1, random_state=42)
classifier_tuned.fit(X_train_preprocessed, y_train)
print("best threshold: ", classifier_tuned.best_threshold_)
y_pred = classifier_tuned.predict(X_test_preprocessed)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

Pt_x = PowerTransformer(method="yeo-johnson")

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

pipeline_final = Pipeline(
    steps=[
        ("column_transformer", transformer),
        ("yeo-johnson", Pt_x),
        ("tuned_classifier", classifier_tuned)
    ]
)

df = pd.read_csv(RAW_DATA_PATH)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
pipeline_final.fit(X_train, y_train)

# saving model
with open( API_PATH / "xgboost_tuned.pkl", "wb") as file:
    pickle.dump(pipeline_final, file)