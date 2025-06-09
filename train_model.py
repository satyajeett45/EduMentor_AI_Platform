# train_model.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE

from config import (
    STUDENT_DATA_PATH, REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)
from preprocessing import preprocess_data

# Load data
df = pd.read_csv(STUDENT_DATA_PATH)
df = df.sample(20000, random_state=42)
# Split features and targets
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y_reg = df[TARGET_COLUMN]
y_clf = df['is_at_risk']

# Preprocess input features
X_processed = preprocess_data(X, fit=True)

# Train-test split for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_processed, y_reg, test_size=0.2, random_state=42)

# ---------- Regression Models ----------
regressors = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor()
}

print("\n----- Regression Models (risk_score) -----")
best_regressor, best_r2 = None, -1
for name, model in regressors.items():
    model.fit(X_reg_train, y_reg_train)
    y_pred = model.predict(X_reg_test)
    r2 = r2_score(y_reg_test, y_pred)
    mae = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    print(f"{name}: R2={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    if r2 > best_r2:
        best_r2 = r2
        best_regressor = model

# Save best regressor
joblib.dump(best_regressor, REGRESSOR_MODEL_PATH)

# ---------- Classification Models ----------
# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_clf_resampled, y_clf_resampled = smote.fit_resample(X_processed, y_clf)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf_resampled, y_clf_resampled, test_size=0.2, random_state=42)

classifiers = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'DecisionTreeClassifier': DecisionTreeClassifier(class_weight='balanced'),
    'RandomForestClassifier': RandomForestClassifier(class_weight='balanced'),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier()
}

print("\n----- Classification Models (is_at_risk) -----")
best_classifier, best_acc = None, 0
for name, model in classifiers.items():
    model.fit(X_clf_train, y_clf_train)
    y_pred = model.predict(X_clf_test)
    acc = accuracy_score(y_clf_test, y_pred)
    print(f"{name}: Accuracy={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_classifier = model

# Save best classifier
joblib.dump(best_classifier, CLASSIFIER_MODEL_PATH)
