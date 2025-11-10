# Simplified PCA Analysis Notebook

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Essential imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePCAAnalyzer:
    """Simplified PCA analyzer for comprehensive analysis"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_with_pca(self, X, y, property_name, n_components=5):
        """Perform PCA analysis and model training"""
        
        logger.info(f"🔬 Analyzing {property_name} with PCA...")
        
        # Standard scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA transformation
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(random_state=42)),
            'GaussianProcess': make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), random_state=42)
            )
        }
        
        results = {}
        
        # Test with original features
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_percentage_error')
            mape = -cv_scores.mean()
            results[f'{name}_Original'] = mape
            logger.info(f"{name} (Original): MAPE = {mape:.4f}")
        
        # Test with PCA features
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_pca, y, cv=5, scoring='neg_mean_absolute_percentage_error')
            mape = -cv_scores.mean()
            results[f'{name}_PCA'] = mape
            logger.info(f"{name} (PCA): MAPE = {mape:.4f}")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda k: results[k])
        best_mape = results[best_model_name]
        
        logger.info(f"🏆 Best model for {property_name}: {best_model_name} (MAPE: {best_mape:.4f})")
        
        # Train best model on full dataset
        if 'PCA' in best_model_name:
            model_name = best_model_name.replace('_PCA', '')
            best_model = models[model_name]
            best_model.fit(X_pca, y)
            features_used = X_pca
            transformer = (scaler, pca)
        else:
            model_name = best_model_name.replace('_Original', '')
            best_model = models[model_name]
            best_model.fit(X_scaled, y)
            features_used = X_scaled
            transformer = (scaler, None)
        
        self.results[property_name] = {
            'model': best_model,
            'model_name': best_model_name,
            'mape': best_mape,
            'transformer': transformer,
            'pca_variance': pca.explained_variance_ratio_.sum(),
            'all_results': results
        }
        
        return best_model, best_model_name, best_mape
    
    def create_predictions(self, test_features, property_index):
        """Create predictions for test data"""
        property_name = f'BlendProperty{property_index}'
        
        if property_name not in self.results:
            logger.error(f"No model trained for {property_name}")
            return None
        
        result = self.results[property_name]
        model = result['model']
        scaler, pca = result['transformer']
        
        # Transform test features
        test_scaled = scaler.transform(test_features)
        if pca is not None:
            test_transformed = pca.transform(test_scaled)
        else:
            test_transformed = test_scaled
        
        # Make predictions
        predictions = model.predict(test_transformed)
        
        logger.info(f"✅ Predictions created for {property_name}")
        logger.info(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        return predictions

print("✅ Simplified PCA Analyzer ready!")
