"""
Pipeline V3 - Smart Feature Engineering
Strategy: Proper Date and Farm_ID engineering, target encoding with smoothing,
meaningful interactions, and V1's successful model parameters.
Performance History:
- V1: CV 4.1271 → Submit 4.17495 (Gap: 0.048)
- V2: CV 4.2482 (worse due to dropped Date/Farm_ID)
- V3: Expected improvement through better feature engineering
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
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

X_full = train.drop(columns=[TARGET_COL, ID_COL]).copy()
y_full = train[TARGET_COL].copy()
X_test_original = test.drop(columns=[ID_COL]).copy()
test_ids = test[ID_COL]

def extract_date_features(df, date_col='Date'):
    if date_col not in df.columns:
        return df
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    df['date_month'] = df[date_col].dt.month
    df['date_quarter'] = df[date_col].dt.quarter
    df['date_day_of_week'] = df[date_col].dt.dayofweek
    df['date_day_of_year'] = df[date_col].dt.dayofyear
    df['date_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    
    df['date_month_sin'] = np.sin(2 * np.pi * df['date_month'] / 12)
    df['date_month_cos'] = np.cos(2 * np.pi * df['date_month'] / 12)
    
    df['date_season'] = df['date_month'].apply(
        lambda x: 0 if x in [12, 1, 2] else
                  1 if x in [3, 4, 5] else
                  2 if x in [6, 7, 8] else
                  3
    )
    
    df = df.drop(columns=[date_col])
    
    return df

X_full = extract_date_features(X_full, 'Date')
X_test = extract_date_features(X_test_original, 'Date')

if 'Farm_ID' in X_full.columns:
    global_mean = y_full.mean()
    
    farm_means = train.groupby('Farm_ID')[TARGET_COL].agg(['mean', 'count']).reset_index()
    
    smoothing = 10
    farm_means['farm_target_encoded'] = (
        (farm_means['mean'] * farm_means['count'] + global_mean * smoothing) /
        (farm_means['count'] + smoothing)
    )
    
    farm_encoding_map = dict(zip(farm_means['Farm_ID'], farm_means['farm_target_encoded']))
    
    X_full['farm_target_encoded'] = X_full['Farm_ID'].map(farm_encoding_map).fillna(global_mean)
    X_test['farm_target_encoded'] = X_test['Farm_ID'].map(farm_encoding_map).fillna(global_mean)
    
    le_farm = LabelEncoder()
    all_farms = pd.concat([X_full['Farm_ID'], X_test['Farm_ID']])
    le_farm.fit(all_farms.astype(str))
    
    X_full['farm_label_encoded'] = le_farm.transform(X_full['Farm_ID'].astype(str))
    X_test['farm_label_encoded'] = le_farm.transform(X_test['Farm_ID'].astype(str))
    
    X_full = X_full.drop(columns=['Farm_ID'])
    X_test = X_test.drop(columns=['Farm_ID'])

interactions_created = 0

if 'Age_Months' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['age_x_weight'] = X_full['Age_Months'] * X_full['Weight_kg']
    X_test['age_x_weight'] = X_test['Age_Months'] * X_test['Weight_kg']
    interactions_created += 1

if 'Parity' in X_full.columns and 'Age_Months' in X_full.columns:
    X_full['parity_x_age'] = X_full['Parity'] * X_full['Age_Months']
    X_test['parity_x_age'] = X_test['Parity'] * X_test['Age_Months']
    interactions_created += 1

if 'Feed_Quantity_kg' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['feed_per_kg'] = X_full['Feed_Quantity_kg'] / (X_full['Weight_kg'] + 1)
    X_test['feed_per_kg'] = X_test['Feed_Quantity_kg'] / (X_test['Weight_kg'] + 1)
    interactions_created += 1

if 'Water_Intake_L' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['water_per_kg'] = X_full['Water_Intake_L'] / (X_full['Weight_kg'] + 1)
    X_test['water_per_kg'] = X_test['Water_Intake_L'] / (X_test['Weight_kg'] + 1)
    interactions_created += 1

if 'Temperature_Celsius' in X_full.columns and 'Humidity_Percent' in X_full.columns:
    X_full['temp_humidity_index'] = X_full['Temperature_Celsius'] * X_full['Humidity_Percent'] / 100
    X_test['temp_humidity_index'] = X_test['Temperature_Celsius'] * X_test['Humidity_Percent'] / 100
    interactions_created += 1

if 'farm_label_encoded' in X_full.columns and 'date_month' in X_full.columns:
    X_full['farm_month_interaction'] = X_full['farm_label_encoded'] * X_full['date_month']
    X_test['farm_month_interaction'] = X_test['farm_label_encoded'] * X_test['date_month']
    interactions_created += 1

numeric_features = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_full.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([X_full[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    
    X_full[col] = le.transform(X_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    
    numeric_features.append(col)

imputer = SimpleImputer(strategy='median')
X_full[numeric_features] = imputer.fit_transform(X_full[numeric_features])
X_test[numeric_features] = imputer.transform(X_test[numeric_features])

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

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_cv_scores = cross_val_score(
    xgb_model, X_full_scaled, y_full,
    cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
)
xgb_rmse = -xgb_cv_scores.mean()
xgb_std = xgb_cv_scores.std()

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=15,
    max_features=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_cv_scores = cross_val_score(
    rf_model, X_full_scaled, y_full,
    cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
)
rf_rmse = -rf_cv_scores.mean()
rf_std = rf_cv_scores.std()

base_estimators = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('ridge', Ridge(alpha=10, random_state=RANDOM_STATE))
]

meta_learner = Ridge(alpha=5, random_state=RANDOM_STATE)

stacking = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

stacking_cv_scores = cross_val_score(
    stacking, X_full_scaled, y_full,
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

final_model.fit(X_full_scaled, y_full)

test_predictions = final_model.predict(X_test_scaled)

print(f"\nPrediction Statistics:")
print(f"  Mean:   {test_predictions.mean():.4f}")
print(f"  Median: {np.median(test_predictions):.4f}")
print(f"  Std:    {test_predictions.std():.4f}")
print(f"  Min:    {test_predictions.min():.4f}")
print(f"  Max:    {test_predictions.max():.4f}")

print("\n[STEP 11] Creating Submission File...")

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: test_predictions
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = SUBMISSION_DIR / f'submission_v3_{best_model_name.lower()}_{timestamp}.csv'
submission.to_csv(submission_filename, index=False)

print(f"Submission saved: {submission_filename.name}")
