"""
Improved Model - Strategy A+B Implementation
Strategy A: Remove redundant features to prevent data leakage
Strategy B: Deep optimization of top features with enhanced transformations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, 
                               StackingRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from scipy import stats
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Will use alternatives.")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('../data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path('../models')
SUBMISSION_DIR = Path('../submissions')

for dir_path in [PROCESSED_DATA_DIR, MODEL_DIR, SUBMISSION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW_DATA_DIR / 'cattle_data_train.csv')
test = pd.read_csv(RAW_DATA_DIR / 'cattle_data_test.csv')

target_col = 'Milk_Yield_L'
id_col = 'Cattle_ID'

test_ids = test[id_col]

REDUNDANT_FEATURES = [
    'Feed_Quantity_lb',
    'Previous_Week_Avg_Yield'
]

for feat in REDUNDANT_FEATURES:
    if feat in train.columns:
        train = train.drop(columns=[feat])
        if feat in test.columns:
            test = test.drop(columns=[feat])

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if id_col in numeric_features:
    numeric_features.remove(id_col)
if target_col in numeric_features:
    numeric_features.remove(target_col)

categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()
if id_col in categorical_features:
    categorical_features.remove(id_col)

X = train[numeric_features + categorical_features].copy()
y = train[target_col].copy()

X_test = test[numeric_features + categorical_features].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

missing_stats = {}

for feature in numeric_features:
    if X_train[feature].isnull().sum() > 0:
        skewness = X_train[feature].skew()
        if abs(skewness) > 1:
            fill_value = X_train[feature].median()
        else:
            fill_value = X_train[feature].mean()
        missing_stats[feature] = fill_value
        X_train[feature].fillna(fill_value, inplace=True)
        X_val[feature].fillna(fill_value, inplace=True)
        X_test[feature].fillna(fill_value, inplace=True)

for feature in categorical_features:
    if X_train[feature].isnull().sum() > 0:
        mode_value = X_train[feature].mode()[0] if len(X_train[feature].mode()) > 0 else 'Unknown'
        missing_stats[feature] = mode_value
        X_train[feature].fillna(mode_value, inplace=True)
        X_val[feature].fillna(mode_value, inplace=True)
        X_test[feature].fillna(mode_value, inplace=True)

outlier_count = 0
for feature in numeric_features:
    if feature in X_train.columns:
        Q1 = X_train[feature].quantile(0.25)
        Q3 = X_train[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        if (X_train[feature] < lower_bound).sum() + (X_train[feature] > upper_bound).sum() > 0:
            X_train[feature] = X_train[feature].clip(lower_bound, upper_bound)
            X_val[feature] = X_val[feature].clip(lower_bound, upper_bound)
            X_test[feature] = X_test[feature].clip(lower_bound, upper_bound)
            outlier_count += 1

label_encoders = {}

for cat_feature in categorical_features:
    le = LabelEncoder()
    X_train[cat_feature] = le.fit_transform(X_train[cat_feature].astype(str))
    X_val[cat_feature] = X_val[cat_feature].apply(
        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
    )
    X_test[cat_feature] = X_test[cat_feature].apply(
        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
    )
    label_encoders[cat_feature] = le

corr_with_target = X_train.corrwith(y_train).abs().sort_values(ascending=False)
top_3_features = corr_with_target.head(3).index.tolist()
top_5_features = corr_with_target.head(5).index.tolist()
top_10_features = corr_with_target.head(10).index.tolist()
top_15_features = corr_with_target.head(15).index.tolist()

top3_transform_count = 0

for feat in top_3_features:
    if feat in X_train.columns:
        X_train[f'{feat}_squared'] = X_train[feat] ** 2
        X_val[f'{feat}_squared'] = X_val[feat] ** 2
        X_test[f'{feat}_squared'] = X_test[feat] ** 2
        top3_transform_count += 1
        
        X_train[f'{feat}_cubed'] = X_train[feat] ** 3
        X_val[f'{feat}_cubed'] = X_val[feat] ** 3
        X_test[f'{feat}_cubed'] = X_test[feat] ** 3
        top3_transform_count += 1
        
        if (X_train[feat] >= 0).all():
            X_train[f'{feat}_sqrt'] = np.sqrt(X_train[feat])
            X_val[f'{feat}_sqrt'] = np.sqrt(X_val[feat])
            X_test[f'{feat}_sqrt'] = np.sqrt(X_test[feat])
            top3_transform_count += 1
            
            X_train[f'{feat}_fourth'] = X_train[feat] ** 4
            X_val[f'{feat}_fourth'] = X_val[feat] ** 4
            X_test[f'{feat}_fourth'] = X_test[feat] ** 4
            top3_transform_count += 1
            
            if (X_train[feat] > 0).all():
                X_train[f'{feat}_log'] = np.log1p(X_train[feat])
                X_val[f'{feat}_log'] = np.log1p(X_val[feat])
                X_test[f'{feat}_log'] = np.log1p(X_test[feat])
                top3_transform_count += 1
                
                X_train[f'{feat}_log2'] = np.log1p(X_train[feat] ** 2)
                X_val[f'{feat}_log2'] = np.log1p(X_val[feat] ** 2)
                X_test[f'{feat}_log2'] = np.log1p(X_test[feat] ** 2)
                top3_transform_count += 1

top3_interaction_count = 0

for i in range(len(top_3_features)):
    for j in range(i+1, len(top_3_features)):
        feat1, feat2 = top_3_features[i], top_3_features[j]
        if feat1 in X_train.columns and feat2 in X_train.columns:
            new_feature = f'{feat1}_x_{feat2}'
            X_train[new_feature] = X_train[feat1] * X_train[feat2]
            X_val[new_feature] = X_val[feat1] * X_val[feat2]
            X_test[new_feature] = X_test[feat1] * X_test[feat2]
            top3_interaction_count += 1
            
            if (X_train[feat2].abs() > 1e-6).all():
                new_feature = f'{feat1}_div_{feat2}'
                X_train[new_feature] = X_train[feat1] / (X_train[feat2] + 1e-6)
                X_val[new_feature] = X_val[feat1] / (X_val[feat2] + 1e-6)
                X_test[new_feature] = X_test[feat1] / (X_test[feat2] + 1e-6)
                top3_interaction_count += 1
            
            new_feature = f'{feat1}_x_{feat2}_sq'
            X_train[new_feature] = X_train[feat1] * (X_train[feat2] ** 2)
            X_val[new_feature] = X_val[feat1] * (X_val[feat2] ** 2)
            X_test[new_feature] = X_test[feat1] * (X_test[feat2] ** 2)
            top3_interaction_count += 1

interaction_count = 0

for i in range(min(5, len(top_10_features))):
    for j in range(i+1, min(5, len(top_10_features))):
        if i < 3 and j < 3:
            continue
        feat1, feat2 = top_10_features[i], top_10_features[j]
        if feat1 in X_train.columns and feat2 in X_train.columns:
            new_feature = f'{feat1}_x_{feat2}'
            X_train[new_feature] = X_train[feat1] * X_train[feat2]
            X_val[new_feature] = X_val[feat1] * X_val[feat2]
            X_test[new_feature] = X_test[feat1] * X_test[feat2]
            interaction_count += 1

for i in range(5, min(10, len(top_15_features))):
    for j in range(i+1, min(i+2, len(top_15_features))):
        feat1, feat2 = top_15_features[i], top_15_features[j]
        if feat1 in X_train.columns and feat2 in X_train.columns:
            new_feature = f'{feat1}_x_{feat2}'
            X_train[new_feature] = X_train[feat1] * X_train[feat2]
            X_val[new_feature] = X_val[feat1] * X_val[feat2]
            X_test[new_feature] = X_test[feat1] * X_test[feat2]
            interaction_count += 1

poly_count = 0
for feat in top_10_features[3:7]:
    if feat in X_train.columns:
        X_train[f'{feat}_squared'] = X_train[feat] ** 2
        X_val[f'{feat}_squared'] = X_val[feat] ** 2
        X_test[f'{feat}_squared'] = X_test[feat] ** 2
        poly_count += 1
        
        if (X_train[feat] >= 0).all():
            X_train[f'{feat}_sqrt'] = np.sqrt(X_train[feat])
            X_val[f'{feat}_sqrt'] = np.sqrt(X_val[feat])
            X_test[f'{feat}_sqrt'] = np.sqrt(X_test[feat])
            poly_count += 1
            
            if (X_train[feat] > 0).all():
                X_train[f'{feat}_log'] = np.log1p(X_train[feat])
                X_val[f'{feat}_log'] = np.log1p(X_val[feat])
                X_test[f'{feat}_log2'] = np.log1p(X_test[feat] ** 2)
                poly_count += 1

for df in [X_train, X_val, X_test]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['numeric_mean'] = df[numeric_cols].mean(axis=1)
    df['numeric_std'] = df[numeric_cols].std(axis=1)
    df['numeric_max'] = df[numeric_cols].max(axis=1)
    df['numeric_min'] = df[numeric_cols].min(axis=1)

total_new_features = top3_transform_count + top3_interaction_count + interaction_count + poly_count + 4

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2
    }

trained_models = {}
all_results = []

if HAS_XGBOOST:
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=2,
            gamma=0.05,
            max_delta_step=1,
            random_state=RANDOM_STATE,
            device='cuda:0',
            tree_method='hist',
            n_jobs=-1
        )
    except:
        xgb_model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=2,
            gamma=0.05,
            max_delta_step=1,
            random_state=RANDOM_STATE,
            device='cpu',
            n_jobs=-1
        )
    
    xgb_model.fit(X_train_scaled, y_train)
    xgb_results = evaluate_model(xgb_model, X_train_scaled, y_train, X_val_scaled, y_val, "XGBoost")
    all_results.append(xgb_results)
    trained_models['xgb'] = xgb_model

base_learners = [
    ('ridge', Ridge(alpha=30, random_state=RANDOM_STATE)),
    ('lasso', Lasso(alpha=0.05, random_state=RANDOM_STATE, max_iter=10000)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=30,
                                  max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.05, max_iter=500,
                         early_stopping=True, random_state=RANDOM_STATE)),
]

if HAS_XGBOOST:
    try:
        base_learners.append((
            'xgb1', xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.3, reg_lambda=1.5,
                random_state=RANDOM_STATE, device='cuda:0', tree_method='hist', n_jobs=-1
            )
        ))
        base_learners.append((
            'xgb2', xgb.XGBRegressor(
                n_estimators=600, max_depth=5, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.9, reg_alpha=0.4, reg_lambda=2.0,
                random_state=RANDOM_STATE+1, device='cuda:0', tree_method='hist', n_jobs=-1
            )
        ))
    except:
        base_learners.append((
            'xgb1', xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.3, reg_lambda=1.5,
                random_state=RANDOM_STATE, device='cpu', n_jobs=-1
            )
        ))
        base_learners.append((
            'xgb2', xgb.XGBRegressor(
                n_estimators=600, max_depth=5, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.9, reg_alpha=0.4, reg_lambda=2.0,
                random_state=RANDOM_STATE+1, device='cpu', n_jobs=-1
            )
        ))

if HAS_XGBOOST:
    try:
        meta_learner = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
            device='cuda:0', tree_method='hist', n_jobs=-1
        )
    except:
        meta_learner = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
            device='cpu', n_jobs=-1
        )
else:
    meta_learner = Ridge(alpha=3)

stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

stacking_model.fit(X_train_scaled, y_train)
stacking_results = evaluate_model(stacking_model, X_train_scaled, y_train, X_val_scaled, y_val, "Stacking")
all_results.append(stacking_results)
trained_models['stacking'] = stacking_model

results_df = pd.DataFrame(all_results).sort_values('val_rmse')

best_model_name = results_df.iloc[0]['model_name']
best_val_rmse = results_df.iloc[0]['val_rmse']

target_rmse = 4.15735

X_full = pd.concat([X_train_scaled, X_val_scaled])
y_full = pd.concat([y_train, y_val])

if best_model_name == 'Stacking':
    final_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
elif best_model_name == 'XGBoost':
    try:
        final_model = xgb.XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.85, colsample_bylevel=0.8,
            reg_alpha=0.3, reg_lambda=1.5, min_child_weight=2,
            gamma=0.05, max_delta_step=1, random_state=RANDOM_STATE,
            device='cuda:0', tree_method='hist', n_jobs=-1
        )
    except:
        final_model = xgb.XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.85, colsample_bylevel=0.8,
            reg_alpha=0.3, reg_lambda=1.5, min_child_weight=2,
            gamma=0.05, max_delta_step=1, random_state=RANDOM_STATE,
            n_jobs=-1
        )
else:
    best_model_key = 'stacking' if 'stacking' in trained_models else 'xgb'
    final_model = trained_models[best_model_key]

final_model.fit(X_full, y_full)

test_predictions = final_model.predict(X_test_scaled)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame({
    id_col: test_ids,
    target_col: test_predictions
})

model_tag = 'optimized_AB'
submission_filename = SUBMISSION_DIR / f'improved_submission_{model_tag}_{timestamp}.csv'
submission.to_csv(submission_filename, index=False)

print(f"Submission saved: {submission_filename.name}")

model_filename = MODEL_DIR / f'best_model_{model_tag}_{timestamp}.pkl'
joblib.dump(final_model, model_filename)
