"""
Improved Pipeline V8 - Conservative Optimization
Performance History:
- V5: CV 4.1193 → Submit 4.16787 (Gap: 0.048)
- V6: CV 4.1186 → Submit 4.16585 (Gap: 0.047) - Best
- V7: CV 4.1209 → Not submitted (over-regularized)

Strategy:
- Based on V6's proven 64-feature structure
- Micro-tune regularization parameters
- Add V7's best features (lactation_curve, body_condition)
- Increase n_estimators for better convergence
- Fine-tune stacking meta-learner
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
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('../data/raw')
SUBMISSION_DIR = Path('../submissions')
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Pipeline V8 - Conservative Optimization Based on V6")
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
    df['date_week_of_year'] = df[date_col].dt.isocalendar().week
    df['date_is_weekend'] = (df['date_day_of_week'] >= 5).astype(int)
    df['date_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['date_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    
    # Cyclical encoding
    df['date_month_sin'] = np.sin(2 * np.pi * df['date_month'] / 12)
    df['date_month_cos'] = np.cos(2 * np.pi * df['date_month'] / 12)
    df['date_quarter_sin'] = np.sin(2 * np.pi * df['date_quarter'] / 4)
    df['date_quarter_cos'] = np.cos(2 * np.pi * df['date_quarter'] / 4)
    df['date_week_sin'] = np.sin(2 * np.pi * df['date_week_of_year'] / 52)
    df['date_week_cos'] = np.cos(2 * np.pi * df['date_week_of_year'] / 52)
    
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
    
    farm_diversity = train.groupby('Farm_ID')[ID_COL].nunique().to_dict()
    X_full['farm_diversity'] = X_full['Farm_ID'].map(farm_diversity)
    X_test['farm_diversity'] = X_test['Farm_ID'].map(farm_diversity).fillna(X_full['farm_diversity'].median())
    
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
    X_full['age_weight_ratio'] = X_full['Age_Months'] / (X_full['Weight_kg'] + 1e-5)
    X_test['age_weight_ratio'] = X_test['Age_Months'] / (X_test['Weight_kg'] + 1e-5)

if 'Parity' in X_full.columns and 'Age_Months' in X_full.columns:
    X_full['parity_x_age'] = X_full['Parity'] * X_full['Age_Months']
    X_test['parity_x_age'] = X_test['Parity'] * X_test['Age_Months']
    X_full['parity_age_ratio'] = X_full['Parity'] / (X_full['Age_Months'] + 1e-5)
    X_test['parity_age_ratio'] = X_test['Parity'] / (X_test['Age_Months'] + 1e-5)

if 'Temperature_Celsius' in X_full.columns and 'Humidity_Percent' in X_full.columns:
    X_full['heat_stress'] = X_full['Temperature_Celsius'] * X_full['Humidity_Percent'] / 100
    X_test['heat_stress'] = X_test['Temperature_Celsius'] * X_test['Humidity_Percent'] / 100
    X_full['temp_squared'] = X_full['Temperature_Celsius'] ** 2
    X_test['temp_squared'] = X_test['Temperature_Celsius'] ** 2
    X_full['humidity_squared'] = X_full['Humidity_Percent'] ** 2
    X_test['humidity_squared'] = X_test['Humidity_Percent'] ** 2

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
if all(col in X_full.columns for col in ['Age_Months', 'Weight_kg', 'Parity']):
    X_full['age_weight_parity'] = X_full['Age_Months'] * X_full['Weight_kg'] * (X_full['Parity'] + 1)
    X_test['age_weight_parity'] = X_test['Age_Months'] * X_test['Weight_kg'] * (X_test['Parity'] + 1)

if all(col in X_full.columns for col in ['Feed_Quantity_kg', 'Feed_Energy_MJ', 'Weight_kg']):
    X_full['nutrition_density'] = (X_full['Feed_Quantity_kg'] * X_full['Feed_Energy_MJ']) / (X_full['Weight_kg'] + 1e-5)
    X_test['nutrition_density'] = (X_test['Feed_Quantity_kg'] * X_test['Feed_Energy_MJ']) / (X_test['Weight_kg'] + 1e-5)

if all(col in X_full.columns for col in ['Temperature_Celsius', 'Humidity_Percent', 'date_season']):
    X_full['seasonal_stress'] = X_full['heat_stress'] * (X_full['date_season'] + 1)
    X_test['seasonal_stress'] = X_test['heat_stress'] * (X_test['date_season'] + 1)

if 'Age_Months' in X_full.columns and 'date_season' in X_full.columns:
    X_full['age_x_season'] = X_full['Age_Months'] * X_full['date_season']
    X_test['age_x_season'] = X_test['Age_Months'] * X_test['date_season']

if 'Parity' in X_full.columns and 'date_month' in X_full.columns:
    X_full['parity_x_month'] = X_full['Parity'] * X_full['date_month']
    X_test['parity_x_month'] = X_test['Parity'] * X_test['date_month']
if all(col in X_full.columns for col in ['Parity', 'date_month']):
    X_full['lactation_curve'] = X_full['Parity'] * np.exp(-0.05 * X_full['date_month'])
    X_test['lactation_curve'] = X_test['Parity'] * np.exp(-0.05 * X_test['date_month'])

if all(col in X_full.columns for col in ['Weight_kg', 'Age_Months', 'Feed_Quantity_kg']):
    X_full['body_condition'] = X_full['Weight_kg'] / (X_full['Age_Months'] + 1) * X_full['Feed_Quantity_kg']
    X_test['body_condition'] = X_test['Weight_kg'] / (X_test['Age_Months'] + 1) * X_test['Feed_Quantity_kg']

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

for feat in ['Age_Months', 'Weight_kg', 'Parity', 'Feed_Quantity_kg']:
    if feat in X_full.columns:
        X_full[f'{feat}_squared'] = X_full[feat] ** 2
        X_test[f'{feat}_squared'] = X_test[feat] ** 2

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

def evaluate_model(model, X, y, name):
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse = -scores.mean()
    return rmse, model

models_perf = {}

xgb_model = xgb.XGBRegressor(
    n_estimators=850,
    max_depth=5,
    learning_rate=0.0175,
    subsample=0.70,
    colsample_bytree=0.70,
    reg_alpha=1.35,
    reg_lambda=3.2,
    min_child_weight=5,
    gamma=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist'
)
xgb_rmse, _ = evaluate_model(xgb_model, X_full_scaled, y, "XGBoost")
models_perf['XGBoost'] = xgb_rmse

lgb_model = lgb.LGBMRegressor(
    n_estimators=850,
    max_depth=5,
    learning_rate=0.0175,
    num_leaves=27,
    subsample=0.70,
    colsample_bytree=0.70,
    reg_alpha=1.35,
    reg_lambda=3.2,
    min_child_weight=5,
    min_child_samples=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)
lgb_rmse, _ = evaluate_model(lgb_model, X_full_scaled, y, "LightGBM")
models_perf['LightGBM'] = lgb_rmse
et_model = ExtraTreesRegressor(
    n_estimators=350,
    max_depth=14,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
et_rmse, _ = evaluate_model(et_model, X_full_scaled, y, "ExtraTrees")
models_perf['ExtraTrees'] = et_rmse

base_learners = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('et', et_model)
]

meta_learner = Ridge(alpha=13.0)

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

stack_rmse, _ = evaluate_model(stacking_model, X_full_scaled, y, "Stacking")
models_perf['Stacking'] = stack_rmse

best_name = min(models_perf, key=models_perf.get)
best_cv = models_perf[best_name]

print(f"{best_name} selected as final model (CV: {best_cv:.4f})")

if best_name == 'XGBoost':
    final_model = xgb_model
elif best_name == 'LightGBM':
    final_model = lgb_model
elif best_name == 'ExtraTrees':
    final_model = et_model
else:
    final_model = stacking_model

final_model.fit(X_full_scaled, y)
predictions = final_model.predict(X_test_scaled)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'submission_v8_{best_name.lower()}_{timestamp}.csv'
filepath = SUBMISSION_DIR / filename

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: predictions
})

submission.to_csv(filepath, index=False)

print(f"Submission saved: {filename}")
