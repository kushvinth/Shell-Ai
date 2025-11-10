import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import skew, kurtosis
from sklearn.metrics import mean_absolute_percentage_error


import warnings
warnings.filterwarnings('ignore')
print("Loading data...")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# --- Feature Engineering ---
def create_breakthrough_features(df, pca_model=None, scaler=None, fit_transformers=True):
    features = [f'Component{i}_fraction' for i in range(1, 6)]
    features += [f'Component{i}_Property{j}' for i in range(1, 6) for j in range(1, 11)]
    # Enhanced interaction features with non-linear transformations
    for i in range(1, 6):
        for j in range(1, 11):
            df[f'frac{i}_prop{j}'] = df[f'Component{i}_fraction'] * df[f'component{i}_Property{j}']
            df[f'frac{i}_prop{j}_sqrt'] = df[f'Component{i}_fraction'] * np.sqrt(np.abs(df[f'Component{i}_Property{j}']))
            df[f"frac{i}_prop{j}_log"] = df[f'Component{i}_fraction'] * np.log(np.abs((df[f'Component{i}_Property{j}'])))
            df[f'frac{i}_prop{j}_square'] = df[f'Component{i}_fraction'] * (df[f'Component{i}_Property{j}'] ** 2)
            features.extend([f'frac{i}_prop{j}_sqrt', f'frac{i}_prop{j}_square'])
            
            
    
    
    # Weighted features
    for j in range(1, 11):
        prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6)]
        frac_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        # Weighted mean
        df[f'weighted_mean_prop{j}'] = sum(df[f'Component{i}_fraction'] * df[f'Component{i}_Property{j}'] for i in range(1, 6))
        mean = df[f'weighted_mean_prop{j}']
        # Weighted variance
        df[f'weighted_var_prop{j}'] = sum(df[f'Component{i}_fraction'] * (df[f'Component{i}_Property{j}'] - mean) ** 2 for i in range(1, 6))
        # Harmonic mean
        safe_props = [np.maximum(df[f'Component{i}_Property{j}'], 1e-6) for i in range(1, 6)]
        harmonic_mean = sum(df[f'Component{i}_fraction'] / safe_props[i-1] for i in range(1, 6))
        df[f'harmonic_mean_prop{j}'] = 1 / harmonic_mean
        # Geometric mean
        log_geo_mean = sum(df[f'Component{i}_fraction'] * np.log(safe_props[i-1]) for i in range(1, 6))
        df[f'geometric_mean_prop{j}'] = np.exp(log_geo_mean)
        # Dominant property
        frac_array = np.array([df[f'Component{i}_fraction'] for i in range(1, 6)])
        dominant_idx = np.argmax(frac_array, axis=0)
        df[f'dominant_prop{j}'] = [df.loc[idx, f'Component{dominant+1}_Property{j}'] for idx, dominant in enumerate(dominant_idx)]
        # Blend balance/diversity
        df[f'blend_balance_prop{j}'] = 1 - df[frac_cols].std(axis=1)
        df[f'blend_diversity_prop{j}'] = df[frac_cols].std(axis=1) / (df[frac_cols].mean(axis=1) + 1e-8)
        # Advanced statistics
        df[f'min_prop{j}'] = df[prop_cols].min(axis=1)
        df[f'max_prop{j}'] = df[prop_cols].max(axis=1)
        df[f'mean_prop{j}'] = df[prop_cols].mean(axis=1)
        df[f'std_prop{j}'] = df[prop_cols].std(axis=1)
        df[f'median_prop{j}'] = df[prop_cols].median(axis=1)
        df[f'skew_prop{j}'] = df[prop_cols].apply(lambda row: skew(row), axis=1)
        df[f'kurtosis_prop{j}'] = df[prop_cols].apply(lambda row: kurtosis(row), axis=1)
        df[f'range_prop{j}'] = df[f'max_prop{j}'] - df[f'min_prop{j}']
        df[f'iqr_prop{j}'] = df[prop_cols].quantile(0.75, axis=1) - df[prop_cols].quantile(0.25, axis=1)
        features.extend([
            f'weighted_mean_prop{j}', f'weighted_var_prop{j}', f'harmonic_mean_prop{j}',
            f'geometric_mean_prop{j}', f'dominant_prop{j}', f'blend_balance_prop{j}',
            f'blend_diversity_prop{j}', f'min_prop{j}', f'max_prop{j}', f'mean_prop{j}',
            f'std_prop{j}', f'median_prop{j}', f'skew_prop{j}', f'kurtosis_prop{j}',
            f'range_prop{j}', f'iqr_prop{j}'
        ])

    # Shell-specific advanced features
    for j in range(1, 11):
        fractions = [df[f'Component{i}_fraction'] for i in range(1, 6)]
        props = [df[f'Component{i}_Property{j}'] for i in range(1, 6)]
        safe_props = [np.maximum(p, 1e-6) for p in props]
        # RON-like blending
        ron_blend = sum(f * (r ** 1.5) for f, r in zip(fractions, safe_props)) ** (1/1.5)
        df[f'ron_like_blend_prop{j}'] = ron_blend
        # Viscosity-like blending
        log_visc_blend = sum(f * np.log(r) for f, r in zip(fractions, safe_props))
        df[f'log_visc_blend_prop{j}'] = log_visc_blend
        # Density-like blending
        density_blend = sum(f * r for f, r in zip(fractions, safe_props))
        df[f'density_blend_prop{j}'] = density_blend
        # Reid vapor pressure-like
        rvp_blend = sum(f * np.exp(r/100) for f, r in zip(fractions, safe_props))
        df[f'rvp_blend_prop{j}'] = rvp_blend
        features.extend([
            f'ron_like_blend_prop{j}', f'log_visc_blend_prop{j}',
            f'density_blend_prop{j}', f'rvp_blend_prop{j}'
        ])

    # Fraction-based advanced features
    frac_cols = [f'Component{i}_fraction' for i in range(1, 6)]
    df['frac_sum'] = df[frac_cols].sum(axis=1)
    df['frac_std'] = df[frac_cols].std(axis=1)
    df['frac_skew'] = df[frac_cols].apply(lambda row: skew(row), axis=1)
    df['frac_kurtosis'] = df[frac_cols].apply(lambda row: kurtosis(row), axis=1)
    df['frac_entropy'] = -sum(df[f'Component{i}_fraction'] * np.log(df[f'Component{i}_fraction'] + 1e-8) for i in range(1, 6))
    df['frac_gini'] = 1 - sum(df[f'Component{i}_fraction'] ** 2 for i in range(1, 6))
    features.extend(['frac_sum', 'frac_std', 'frac_skew', 'frac_kurtosis', 'frac_entropy', 'frac_gini'])

    # PCA features (if fit_transformers)
    prop_features = [f'Component{i}_Property{j}' for i in range(1, 6) for j in range(1, 11)]
    if fit_transformers:
        pca = PCA(n_components=12, random_state=42)
        pca_feats = pca.fit_transform(df[prop_features])
    else:
        pca = pca_model
        pca_feats = pca.transform(df[prop_features])
    for k in range(12):
        df[f'pca_prop_{k+1}'] = pca_feats[:, k]
        features.append(f'pca_prop_{k+1}')
    return df, features, pca

# --- Prepare features ---
print("Creating breakthrough features...")
train, feat_cols, pca_model = create_breakthrough_features(train, fit_transformers=True)
test, _, _ = create_breakthrough_features(test, pca_model=pca_model, fit_transformers=False)

# --- Feature scaling ---
scaler_robust = RobustScaler()
x_train = train[feat_cols]
x_test = test[feat_cols]
x_train_robust = scaler_robust.fit_transform(x_train)
x_test_robust = scaler_robust.transform(x_test)

scaler_standard = StandardScaler()
x_train_standard = scaler_standard.fit_transform(x_train)
x_test_standard = scaler_standard.transform(x_test)

y_train = train[[col for col in train.columns if col not in feat_cols]]
TARGETS = y_train.columns.tolist()

# --- Feature selection ---
print("Performing feature selection...")
selector = SelectFromModel(LGBMRegressor(n_estimators=200, random_state=42, verbose=-1), prefit=False, threshold="median")
x_train_selected = selector.fit_transform(x_train, y_train.iloc[:, 0])
x_test_selected = selector.transform(x_test)
selected_features = [feat_cols[i] for i in range(len(feat_cols)) if selector.get_support()[i]]
print(f"original features: {len(feat_cols)}")
print(f"selected features: {len(selected_features)}")

# --- Cross-validation and ensemble training ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros((x_test.shape[0], len(TARGETS)))
print("Training Breakthrough Ensemble...")
print(f"Features: {len(feat_cols)} (selected: {len(selected_features)})")

for i, target in enumerate(TARGETS):
    print(f"\ntraining for {target}...")
    lgb_oof = np.zeros(x_train.shape[0])
    rf_oof = np.zeros(x_train.shape[0])
    et_oof = np.zeros(x_train.shape[0])
    gb_oof = np.zeros(x_train.shape[0])
    ridge_oof = np.zeros(x_train.shape[0])
    elastic_oof = np.zeros(x_train.shape[0])
    huber_oof = np.zeros(x_train.shape[0])
    lgb_test_preds = np.zeros(x_test.shape[0])
    rf_test_preds = np.zeros(x_test.shape[0])
    et_test_preds = np.zeros(x_test.shape[0])
    gb_test_preds = np.zeros(x_test.shape[0])
    ridge_test_preds = np.zeros(x_test.shape[0])
    elastic_test_preds = np.zeros(x_test.shape[0])
    huber_test_preds = np.zeros(x_test.shape[0])
    for fold, (tr_idx, val_idx) in enumerate(kf.split(x_train)):
        # Model 1: LightGBM
        model_lgb = LGBMRegressor(
            n_estimators=12000, learning_rate=0.0015, random_state=fold,
            num_leaves=31, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.01, reg_lambda=0.01, min_child_samples=20,
            objective='regression_l1')
        model_lgb.fit(
            x_train.iloc[tr_idx], y_train[target].iloc[tr_idx],
            eval_set=[(x_train.iloc[val_idx], y_train[target].iloc[val_idx])],
            callbacks=[early_stopping(stopping_rounds=150), log_evaluation(200)])
        lgb_oof[val_idx] = np.asarray(model_lgb.predict(x_train.iloc[val_idx])).ravel()
        lgb_test_preds += np.asarray(model_lgb.predict(x_test)).ravel() / kf.get_n_splits()
        # Model 2: Random Forest
        model_rf = RandomForestRegressor(n_estimators=800, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=fold, n_jobs=-1)
        model_rf.fit(x_train.iloc[tr_idx], y_train[target].iloc[tr_idx])
        rf_oof[val_idx] = np.asarray(model_rf.predict(x_train.iloc[val_idx])).ravel()
        rf_test_preds += np.asarray(model_rf.predict(x_test)).ravel() / kf.get_n_splits()
        # Model 3: Extra Trees
        model_et = ExtraTreesRegressor(n_estimators=600, max_depth=18, min_samples_split=3, min_samples_leaf=1, random_state=fold, n_jobs=-1)
        model_et.fit(x_train.iloc[tr_idx], y_train[target].iloc[tr_idx])
        et_oof[val_idx] = np.asarray(model_et.predict(x_train.iloc[val_idx])).ravel()
        et_test_preds += np.asarray(model_et.predict(x_test)).ravel() / kf.get_n_splits()
        # Model 4: Gradient Boosting
        model_gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=fold)
        model_gb.fit(x_train.iloc[tr_idx], y_train[target].iloc[tr_idx])
        gb_oof[val_idx] = np.asarray(model_gb.predict(x_train.iloc[val_idx])).ravel()
        gb_test_preds += np.asarray(model_gb.predict(x_test)).ravel() / kf.get_n_splits()
        # Model 5: Ridge (robust scaling)
        model_ridge = Ridge(alpha=0.03, random_state=fold)
        model_ridge.fit(x_train_robust[tr_idx], y_train[target].iloc[tr_idx])
        ridge_oof[val_idx] = np.asarray(model_ridge.predict(x_train_robust[val_idx])).ravel()
        ridge_test_preds += np.asarray(model_ridge.predict(x_test_robust)).ravel() / kf.get_n_splits()
        # Model 6: Elastic Net (standard scaling)
        model_elastic = ElasticNet(alpha=0.008, l1_ratio=0.3, random_state=fold, max_iter=2000)
        model_elastic.fit(x_train_standard[tr_idx], y_train[target].iloc[tr_idx])
        elastic_oof[val_idx] = np.asarray(model_elastic.predict(x_train_standard[val_idx])).ravel()
        elastic_test_preds += np.asarray(model_elastic.predict(x_test_standard)).ravel() / kf.get_n_splits()
        # Model 7: Huber (robust scaling)
        model_huber = HuberRegressor(alpha=0.01, epsilon=1.35)
        model_huber.fit(x_train_robust[tr_idx], y_train[target].iloc[tr_idx])
        huber_oof[val_idx] = np.asarray(model_huber.predict(x_train_robust[val_idx])).ravel()
        huber_test_preds += np.asarray(model_huber.predict(x_test_robust)).ravel() / kf.get_n_splits()
    # Calculate MAPE for each model
    lgb_mape = mean_absolute_percentage_error(y_train[target], lgb_oof)
    rf_mape = mean_absolute_percentage_error(y_train[target], rf_oof)
    et_mape = mean_absolute_percentage_error(y_train[target], et_oof)
    gb_mape = mean_absolute_percentage_error(y_train[target], gb_oof)
    ridge_mape = mean_absolute_percentage_error(y_train[target], ridge_oof)
    elastic_mape = mean_absolute_percentage_error(y_train[target], elastic_oof)
    huber_mape = mean_absolute_percentage_error(y_train[target], huber_oof)
    # Exponential weighting
    mape_scores = [lgb_mape, rf_mape, et_mape, gb_mape, ridge_mape, elastic_mape, huber_mape]
    weights = [np.exp(-score * 10) for score in mape_scores]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    # Final predictions
    final_preds[:, i] = (
        weights[0] * lgb_test_preds +
        weights[1] * rf_test_preds +
        weights[2] * et_test_preds +
        weights[3] * gb_test_preds +
        weights[4] * ridge_test_preds +
        weights[5] * elastic_test_preds +
        weights[6] * huber_test_preds
    )
    # Ensemble validation score
    ensemble_oof = (
        weights[0] * lgb_oof +
        weights[1] * rf_oof +
        weights[2] * et_oof +
        weights[3] * gb_oof +
        weights[4] * ridge_oof +
        weights[5] * elastic_oof +
        weights[6] * huber_oof
    )
    ensemble_mape = mean_absolute_percentage_error(y_train[target], ensemble_oof)
    print(f"LightGBM MAPE: {lgb_mape:.4f} (weight: {weights[0]:.3f})")
    print(f"Random Forest MAPE: {rf_mape:.4f} (weight: {weights[1]:.3f})")
    print(f"Extra Trees MAPE: {et_mape:.4f} (weight: {weights[2]:.3f})")
    print(f"Gradient Boosting MAPE: {gb_mape:.4f} (weight: {weights[3]:.3f})")
    print(f"Ridge MAPE: {ridge_mape:.4f} (weight: {weights[4]:.3f})")
    print(f"Elastic Net MAPE: {elastic_mape:.4f} (weight: {weights[5]:.3f})")
    print(f"Huber MAPE: {huber_mape:.4f} (weight: {weights[6]:.3f})")
    print(f"Ensemble MAPE: {ensemble_mape:.4f}")

# --- Submission ---
submission = pd.DataFrame(final_preds, columns=TARGETS)
submission.insert(0, 'ID', test.get('ID', np.arange(1, len(test) + 1)))
submission.to_csv('submission_breakthrough_90plus.csv', index=False)
print("\nSubmission file created: submission_breakthrough_90plus.csv")
print(f"\nBreakthrough Ensemble Summary:")
print(f"Features: {len(feat_cols)} (selected: {len(selected_features)})")
print(f"Cross-validation: 5-fold")
print(f"Models: LightGBM, Random Forest, Extra Trees, Gradient Boosting, Ridge, Elastic Net, Huber")
print(f"Ensemble: Exponential weighting based on validation performance")
print(f"Scaling: Robust and Standard scaling for different models")
print(f"Target: 90+ score with breakthrough features and advanced ensemble")
