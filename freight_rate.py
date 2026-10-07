

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer

from lightgbm import LGBMRegressor

# Load dataset
file_path = 'data/train-test.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=['posted_rate'])
y = df['posted_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# KNN imputer yields lower validation RMSE than simple median fill
num_colms = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
             'distance', 'weight', 'market_index', 'quote_signal']

imputer = KNNImputer(n_neighbors=5)
X_train[num_colms] = imputer.fit_transform(X_train[num_colms])
X_test[num_colms] = imputer.transform(X_test[num_colms])

# Drop extreme outliers skewing loss during training
train_mask = y_train < 15000
X_train = X_train[train_mask]
y_train = y_train[train_mask]


# Feature engineering
def add_time_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df.drop(columns=['date'])

def add_spatial_features(df):
    df = df.copy()
    if 'pickup' in df.columns and 'delivery' in df.columns:
        df['lane_id'] = df['pickup'].astype(str) + '_' + df['delivery'].astype(str)

    if all(col in df.columns for col in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']):
        df['lat_diff'] = df['delivery_lat'] - df['pickup_lat']
        df['lon_diff'] = df['delivery_lon'] - df['pickup_lon']
        df['coord_distance'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)

    if 'distance' in df.columns:
        df['log_distance'] = np.log1p(df['distance'])
    return df

val_df = pd.read_csv('data/validation.csv')

X_train = add_time_features(X_train)
X_test = add_time_features(X_test)
val_df = add_time_features(val_df)

X_train = add_spatial_features(X_train)
X_test = add_spatial_features(X_test)
val_df = add_spatial_features(val_df)

# One-hot encode equipment and set categorical dtypes for route strings
X_train = pd.get_dummies(X_train, columns=['equipment'], drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=['equipment'], drop_first=True, dtype=int)
val_df = pd.get_dummies(val_df, columns=['equipment'], drop_first=True, dtype=int)

for col in ['pickup', 'delivery', 'lane_id']:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        val_df[col] = val_df[col].astype('category')

# Keep feature columns identical across train, test, and validation sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
X_train, val_df = X_train.align(val_df, join='left', axis=1, fill_value=0)

features = [col for col in X_train.columns if col != 'load_id' and X_train[col].dtype != 'object']

X_train_fit = X_train[features]
X_test_fit = X_test[features]

# Quick baseline evaluation
baseline_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
baseline_model.fit(X_train_fit, y_train, eval_set=[(X_test_fit, y_test)], callbacks=[])
y_pred = baseline_model.predict(X_test_fit)

print(f"Baseline LightGBM -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} | MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores, mae_scores = [], []

print("\nRunning 5-fold CV...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_fit)):
    X_tr, X_va = X_train_fit.iloc[tr_idx], X_train_fit.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    cv_model = LGBMRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    cv_model.fit(X_tr, y_tr)
    preds = cv_model.predict(X_va)

    rmse_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    mae_scores.append(mean_absolute_error(y_va, preds))
    print(f"Fold {fold + 1} -> RMSE: {rmse_scores[-1]:.2f} | MAE: {mae_scores[-1]:.2f}")

print(f"\nCV Average RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
print(f"CV Average MAE:  {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")

# Fit final model on full train split
final_model = LGBMRegressor(
    n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
)
final_model.fit(X_train_fit, y_train)

# Save validation set predictions
X_val = val_df[X_train_fit.columns].copy()
val_preds = final_model.predict(X_val)

submission_df = pd.DataFrame({
    'load_id': val_df['load_id'],
    'predicted_rate': np.round(val_preds, 2)
})
submission_df.to_csv('validation_predictions.csv', index=False)
print("\nGenerated validation_predictions.csv successfully.")

# Final test evaluation
test_preds = final_model.predict(X_test_fit)
print(f"\nTest RMSE: ${np.sqrt(mean_squared_error(y_test, test_preds)):,.2f}")
print(f"Test MAE:  ${mean_absolute_error(y_test, test_preds):,.2f}")
print(f"Test R²:   {r2_score(y_test, test_preds):.4f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=test_preds, alpha=0.3, color='#1f77b4')
min_val, max_val = min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
plt.title('Actual vs Predicted Freight Rates')
plt.xlabel('Actual Rate ($)')
plt.ylabel('Predicted Rate ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('rate_prediction_performance.png', dpi=300)
plt.show()

# Process December chart input file
dec_ch_path = 'data/december_chart_inputs.csv'
if not os.path.exists(dec_ch_path) and os.path.exists('data/december-chart-inputs.csv'):
    dec_ch_path = 'data/december-chart-inputs.csv'

dec_df = pd.read_csv(dec_ch_path)
dec_processed = add_spatial_features(dec_df)
dec_processed = add_time_features(dec_processed)
dec_processed = pd.get_dummies(dec_processed)
dec_processed = dec_processed.reindex(columns=X_train_fit.columns, fill_value=0)

for col in X_train_fit.columns:
    if isinstance(X_train_fit[col].dtype, pd.CategoricalDtype):
        dec_processed[col] = dec_processed[col].astype(X_train_fit[col].dtype)

dec_preds = final_model.predict(dec_processed)
dec_df['predicted_rate'] = np.round(dec_preds, 2)
dec_df.to_csv(dec_ch_path, index=False)
print(f"Updated {dec_ch_path} with predictions.")

# Run assessment scoring script
import os

os.system('python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv')
