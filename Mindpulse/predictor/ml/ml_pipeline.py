import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ۱. بارگذاری داده‌ها
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'project2_Konzas.csv')
df = pd.read_csv(csv_path)

# ۲. پاک‌سازی داده و جداسازی ویژگی‌ها
wb_cols = [f'wb{i}' for i in range(1, 15)]
drop_cols = ['Unnamed: 0', 'well'] + wb_cols

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['well']

valid_idx = y.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ۳. پیش‌پردازش
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ۴. تعریف تمامی مدل‌ها
all_models = {
    'Linear Regression': (LinearRegression(), True),
    'Ridge': (Ridge(alpha=10.0), True),
    'Lasso': (Lasso(alpha=0.1), True),
    'ElasticNet': (ElasticNet(alpha=0.1, l1_ratio=0.5), True),
    'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), False),
    'Extra Trees': (ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42), False),
    'XGBoost': (XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42), False),
    'LightGBM': (LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1), False)
}

comparison_results = []

for name, (model, is_scaled) in all_models.items():
    X_tr = X_train_scaled if is_scaled else X_train_imp
    X_te = X_test_scaled if is_scaled else X_test_imp
    
    model.fit(X_tr, y_train)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    
    r2_tr = r2_score(y_train, pred_tr)
    r2_te = r2_score(y_test, pred_te)
    rmse_tr = np.sqrt(mean_squared_error(y_train, pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_test, pred_te))
    
    comparison_results.append({
        'model': name,
        'type': 'Linear' if is_scaled else 'Ensemble',
        'train_r2': round(float(r2_tr), 3),
        'test_r2': round(float(r2_te), 3),
        'train_rmse': round(float(rmse_tr), 3),
        'test_rmse': round(float(rmse_te), 3),
        'overfit_gap': round(float(r2_tr - r2_te), 3)
    })

# ۵. مدل ترکیبی (Hybrid LGBM + ElasticNet)
lgbm_model = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

lgbm_model.fit(X_train_imp, y_train)
elastic_model.fit(X_train_scaled, y_train)

hyb_tr = 0.5 * lgbm_model.predict(X_train_imp) + 0.5 * elastic_model.predict(X_train_scaled)
hyb_te = 0.5 * lgbm_model.predict(X_test_imp) + 0.5 * elastic_model.predict(X_test_scaled)

r2_tr_h = r2_score(y_train, hyb_tr)
r2_te_h = r2_score(y_test, hyb_te)

comparison_results.append({
    'model': 'Hybrid (LGBM + ElasticNet)',
    'type': 'Hybrid',
    'train_r2': round(float(r2_tr_h), 3),
    'test_r2': round(float(r2_te_h), 3),
    'train_rmse': round(float(np.sqrt(mean_squared_error(y_train, hyb_tr))), 3),
    'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, hyb_te))), 3),
    'overfit_gap': round(float(r2_tr_h - r2_te_h), 3)
})

# ۶. ذخیره نتایج مقایسه در JSON و پایپلاین در PKL
json_path = os.path.join(current_dir, 'model_comparison.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, ensure_ascii=False, indent=4)

pkl_path = os.path.join(current_dir, 'wellbeing_hybrid_pipeline.pkl')
joblib.dump({
    'imputer': imputer,
    'scaler': scaler,
    'lgbm': lgbm_model,
    'elastic': elastic_model,
    'feature_names': list(X.columns)
}, pkl_path)

print("✅ مقایسه تمامی مدل‌ها انجام شد و فایل‌های JSON و PKL ذخیره شدند.")