"""
Pipeline V2 - Addressing Overfitting
Strategy: Remove high-cardinality features (Date, Farm_ID), add meaningful interactions,
optimize XGBoost parameters for better generalization, and use feature selection.
Performance History:
- V1: CV 4.1271 → Submit 4.17495 (Gap: 0.048)
- V2: Goal to reduce overfitting gap
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('../data/raw')
SUBMISSION_DIR = Path('../submissions')
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / 'cattle_data_train.csv')
test = pd.read_csv(DATA_DIR / 'cattle_data_test.csv')

TARGET_COL = 'Milk_Yield_L'
ID_COL = 'Cattle_ID'

train.replace('?', np.nan, inplace=True)
test.replace('?', np.nan, inplace=True)

X_full = train.drop(columns=[TARGET_COL, ID_COL])
y_full = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL])
test_ids = test[ID_COL]

FEATURES_TO_DROP = ['Date', 'Farm_ID']
for feat in FEATURES_TO_DROP:
    if feat in X_full.columns:
        X_full = X_full.drop(columns=[feat])
        X_test = X_test.drop(columns=[feat])
        print(f"  ✗ Dropped {feat}")

# Identify feature types
numeric_features = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_full.select_dtypes(include=['object', 'category']).columns.tolist()

interactions_created = 0

if 'Age_Months' in numeric_features and 'Weight_kg' in numeric_features:
    X_full['Age_Weight_Ratio'] = X_full['Age_Months'] / (X_full['Weight_kg'] + 1)
    X_test['Age_Weight_Ratio'] = X_test['Age_Months'] / (X_test['Weight_kg'] + 1)
    numeric_features.append('Age_Weight_Ratio')
    interactions_created += 1

if 'Parity' in numeric_features and 'Age_Months' in numeric_features:
    X_full['Parity_Age_Product'] = X_full['Parity'] * X_full['Age_Months']
    X_test['Parity_Age_Product'] = X_test['Parity'] * X_test['Age_Months']
    numeric_features.append('Parity_Age_Product')
    interactions_created += 1

if 'Rumination_Time_hrs' in numeric_features and 'Feed_Quantity_kg' in numeric_features:
    X_full['Rumination_Feed_Ratio'] = X_full['Rumination_Time_hrs'] / (X_full['Feed_Quantity_kg'] + 1)
    X_test['Rumination_Feed_Ratio'] = X_test['Rumination_Time_hrs'] / (X_test['Feed_Quantity_kg'] + 1)
    numeric_features.append('Rumination_Feed_Ratio')
    interactions_created += 1

if 'Temperature_Celsius' in numeric_features and 'Humidity_Percent' in numeric_features:
    X_full['Temp_Humidity_Index'] = X_full['Temperature_Celsius'] * X_full['Humidity_Percent'] / 100
    X_test['Temp_Humidity_Index'] = X_test['Temperature_Celsius'] * X_test['Humidity_Percent'] / 100
    numeric_features.append('Temp_Humidity_Index')
    interactions_created += 1

for col in numeric_features:
    if X_full[col].isnull().sum() > 0:
        median_val = X_full[col].median()
        X_full[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

label_encoders = {}
for col in categorical_features:
    if X_full[col].isnull().sum() > 0:
        mode_val = X_full[col].mode()[0] if len(X_full[col].mode()) > 0 else 'Unknown'
        X_full[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)
    
    le = LabelEncoder()
    combined = pd.concat([X_full[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    
    X_full[col] = X_full[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    label_encoders[col] = le

scaler = StandardScaler()
X_full_scaled = pd.DataFrame(
    scaler.fit_transform(X_full),
    columns=X_full.columns,
    index=X_full.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

for df in [X_full_scaled, X_test_scaled]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_full_scaled, y_full)

feature_scores = pd.DataFrame({
    'feature': X_full_scaled.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

n_features_to_keep = int(len(X_full_scaled.columns) * 0.7)
selected_features = feature_scores.head(n_features_to_keep)['feature'].tolist()

X_full_selected = X_full_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

xgb_params = {
    'n_estimators': 600,
    'max_depth': 5,
    'learning_rate': 0.02,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'colsample_bylevel': 0.75,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'min_child_weight': 5,
    'gamma': 0.2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_cv_scores = cross_val_score(
    xgb_model, X_full_selected, y_full,
    cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
)
xgb_rmse = -xgb_cv_scores.mean()
xgb_std = xgb_cv_scores.std()

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=20,
    max_features=0.4,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_cv_scores = cross_val_score(
    rf_model, X_full_selected, y_full,
    cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
)
rf_rmse = -rf_cv_scores.mean()
rf_std = rf_cv_scores.std()

base_estimators = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('ridge', Ridge(alpha=15, random_state=RANDOM_STATE))
]

meta_learner = Ridge(alpha=10, random_state=RANDOM_STATE)

stacking = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

stacking_cv_scores = cross_val_score(
    stacking, X_full_selected, y_full,
    cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
)
stacking_rmse = -stacking_cv_scores.mean()
stacking_std = stacking_cv_scores.std()

results = {
    'XGBoost': xgb_rmse,
    'RandomForest': rf_rmse,
    'Stacking': stacking_rmse
}

best_model_name = min(results, key=results.get)
best_cv_rmse = results[best_model_name]

if best_model_name == 'XGBoost':
    final_model = xgb_model
elif best_model_name == 'RandomForest':
    final_model = rf_model
else:
    final_model = stacking

final_model.fit(X_full_selected, y_full)

test_predictions = final_model.predict(X_test_selected)

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: test_predictions
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = SUBMISSION_DIR / f'submission_improved_v2_{best_model_name.lower()}_{timestamp}.csv'
submission.to_csv(submission_filename, index=False)

print(f"Submission saved: {submission_filename.name}")
