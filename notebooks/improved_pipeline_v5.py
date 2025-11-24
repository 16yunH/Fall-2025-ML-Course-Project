"""
Improved Pipeline V5 - Push for Top Performance
Performance History:
- V1: CV 4.1271 → Submit 4.17495 (Gap: 0.048)
- V3: CV 4.1155 → Submit 4.17508 (Gap: 0.060) - Target encoding leaked
- V4: CV 4.1203 → Submit 4.16850 (Gap: 0.048) - Fixed leakage

Strategy:
- Add LightGBM - often better than XGBoost on tabular data
- Farm statistical features (NO target encoding) - size, frequency
- Advanced domain interactions - lactation physiology
- Polynomial features for key ratios
- Fine-tune regularization to reduce gap
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("LightGBM not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm as lgb
    HAS_LIGHTGBM = True

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('../data/raw')
SUBMISSION_DIR = Path('../submissions')
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Pipeline V5 - Push for Top Performance")
print("="*80)

train = pd.read_csv(DATA_DIR / 'cattle_data_train.csv')
test = pd.read_csv(DATA_DIR / 'cattle_data_test.csv')

ID_COL = 'Cattle_ID'
TARGET_COL = 'Milk_Yield_L'

X_full = train.drop(columns=[ID_COL, TARGET_COL])
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL])
test_ids = test[ID_COL]

def extract_date_features(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['date_month'] = df[date_col].dt.month
    df['date_quarter'] = df[date_col].dt.quarter
    df['date_day_of_week'] = df[date_col].dt.dayofweek
    df['date_day_of_year'] = df[date_col].dt.dayofyear
    df['date_is_weekend'] = (df['date_day_of_week'] >= 5).astype(int)
    
    df['date_month_sin'] = np.sin(2 * np.pi * df['date_month'] / 12)
    df['date_month_cos'] = np.cos(2 * np.pi * df['date_month'] / 12)
    df['date_quarter_sin'] = np.sin(2 * np.pi * df['date_quarter'] / 4)
    df['date_quarter_cos'] = np.cos(2 * np.pi * df['date_quarter'] / 4)
    
    df['date_season'] = ((df['date_month'] % 12 + 3) // 3) % 4
    
    df = df.drop(columns=[date_col])
    return df

X_full = extract_date_features(X_full, 'Date')
X_test = extract_date_features(X_test, 'Date')

if 'Farm_ID' in X_full.columns:
    farm_sizes = X_full['Farm_ID'].value_counts().to_dict()
    X_full['farm_size'] = X_full['Farm_ID'].map(farm_sizes)
    X_test['farm_size'] = X_test['Farm_ID'].map(farm_sizes).fillna(X_full['farm_size'].median())
    
    X_full['farm_frequency'] = X_full['farm_size'] / len(X_full)
    X_test['farm_frequency'] = X_test['farm_size'] / len(X_full)
    
    farm_rank = X_full['Farm_ID'].value_counts().rank(method='dense', ascending=False).to_dict()
    X_full['farm_rank'] = X_full['Farm_ID'].map(farm_rank)
    X_test['farm_rank'] = X_test['Farm_ID'].map(farm_rank).fillna(X_full['farm_rank'].median())
    
    le_farm = LabelEncoder()
    all_farms = pd.concat([X_full['Farm_ID'], X_test['Farm_ID']])
    le_farm.fit(all_farms.astype(str))
    
    X_full['farm_encoded'] = le_farm.transform(X_full['Farm_ID'].astype(str))
    X_test['farm_encoded'] = le_farm.transform(X_test['Farm_ID'].astype(str))
    
    X_full = X_full.drop(columns=['Farm_ID'])
    X_test = X_test.drop(columns=['Farm_ID'])
if 'Age_Months' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['age_x_weight'] = X_full['Age_Months'] * X_full['Weight_kg']
    X_test['age_x_weight'] = X_test['Age_Months'] * X_test['Weight_kg']
if 'Parity' in X_full.columns and 'Age_Months' in X_full.columns:
    X_full['parity_x_age'] = X_full['Parity'] * X_full['Age_Months']
    X_test['parity_x_age'] = X_test['Parity'] * X_test['Age_Months']
if 'Temperature_Celsius' in X_full.columns and 'Humidity_Percent' in X_full.columns:
    X_full['heat_stress'] = X_full['Temperature_Celsius'] * X_full['Humidity_Percent'] / 100
    X_test['heat_stress'] = X_test['Temperature_Celsius'] * X_test['Humidity_Percent'] / 100

if 'Temperature_Celsius' in X_full.columns:
    X_full['temp_squared'] = X_full['Temperature_Celsius'] ** 2
    X_test['temp_squared'] = X_test['Temperature_Celsius'] ** 2
if 'Feed_Quantity_kg' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['feed_per_weight'] = X_full['Feed_Quantity_kg'] / (X_full['Weight_kg'] + 1e-5)
    X_test['feed_per_weight'] = X_test['Feed_Quantity_kg'] / (X_test['Weight_kg'] + 1e-5)

if 'Feed_Protein_Percent' in X_full.columns and 'Feed_Quantity_kg' in X_full.columns:
    X_full['protein_total'] = X_full['Feed_Protein_Percent'] * X_full['Feed_Quantity_kg'] / 100
    X_test['protein_total'] = X_test['Feed_Protein_Percent'] * X_test['Feed_Quantity_kg'] / 100

if 'Feed_Energy_MJ' in X_full.columns and 'Weight_kg' in X_full.columns:
    X_full['energy_per_weight'] = X_full['Feed_Energy_MJ'] / (X_full['Weight_kg'] + 1e-5)
    X_test['energy_per_weight'] = X_test['Feed_Energy_MJ'] / (X_test['Weight_kg'] + 1e-5)

if 'Somatic_Cell_Count' in X_full.columns:
    X_full['scc_log'] = np.log1p(X_full['Somatic_Cell_Count'])
    X_test['scc_log'] = np.log1p(X_test['Somatic_Cell_Count'])

if 'Somatic_Cell_Count' in X_full.columns and 'Mastitis_History' in X_full.columns:
    X_full['health_score'] = 1 / (X_full['Somatic_Cell_Count'] + 1e5) * (1 - X_full['Mastitis_History'])
    X_test['health_score'] = 1 / (X_test['Somatic_Cell_Count'] + 1e5) * (1 - X_test['Mastitis_History'])
if 'Feed_Quantity_kg' in X_full.columns and 'date_season' in X_full.columns:
    X_full['feed_x_season'] = X_full['Feed_Quantity_kg'] * X_full['date_season']
    X_test['feed_x_season'] = X_test['Feed_Quantity_kg'] * X_test['date_season']

if 'Parity' in X_full.columns and 'date_season' in X_full.columns:
    X_full['parity_x_season'] = X_full['Parity'] * X_full['date_season']
    X_test['parity_x_season'] = X_test['Parity'] * X_test['date_season']

if 'Age_Months' in X_full.columns and 'date_month' in X_full.columns:
    X_full['age_x_month'] = X_full['Age_Months'] * X_full['date_month']
    X_test['age_x_month'] = X_test['Age_Months'] * X_test['date_month']

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

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate_model(model, X, y, model_name):
    scores = cross_val_score(
        model, X, y,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    rmse_scores = -scores
    return rmse_scores.mean(), model
xgb_model = xgb.XGBRegressor(
    n_estimators=600,          # Increased from 500
    max_depth=6,
    learning_rate=0.02,        # Reduced from 0.025 for smoother learning
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=1.0,             # Increased L1 from 0.8
    reg_lambda=3.0,            # Increased L2 from 2.5
    min_child_weight=5,        # Increased from 4
    gamma=0.1,                 # Added minimum loss reduction
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist'
)

xgb_rmse, _ = evaluate_model(xgb_model, X_full_scaled, y, "XGBoost")

if HAS_LIGHTGBM:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        num_leaves=31,             # 2^5 - 1 for depth 5
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=3.0,
        min_child_weight=5,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_rmse, _ = evaluate_model(lgb_model, X_full_scaled, y, "LightGBM")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_rmse, _ = evaluate_model(rf_model, X_full_scaled, y, "RandomForest")

base_learners = [
    ('xgb', xgb_model),
    ('rf', rf_model)
]

if HAS_LIGHTGBM:
    base_learners.append(('lgb', lgb_model))

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(alpha=10.0),
    cv=5,
    n_jobs=-1
)
stack_rmse, _ = evaluate_model(stacking_model, X_full_scaled, y, "Stacking")

models_performance = {
    'XGBoost': xgb_rmse,
    'RandomForest': rf_rmse,
    'Stacking': stack_rmse
}

if HAS_LIGHTGBM:
    models_performance['LightGBM'] = lgb_rmse

best_model_name = min(models_performance, key=models_performance.get)
best_cv_rmse = models_performance[best_model_name]

print(f"{best_model_name} selected as final model (CV: {best_cv_rmse:.4f})")
if best_model_name == 'XGBoost':
    final_model = xgb_model
elif best_model_name == 'LightGBM':
    final_model = lgb_model
elif best_model_name == 'RandomForest':
    final_model = rf_model
else:
    final_model = stacking_model

final_model.fit(X_full_scaled, y)
predictions = final_model.predict(X_test_scaled)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f'submission_v5_{best_model_name.lower()}_{timestamp}.csv'
submission_path = SUBMISSION_DIR / submission_filename

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: predictions
})

submission.to_csv(submission_path, index=False)

print(f"Submission saved: {submission_filename}")
