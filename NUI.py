import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class MinimalPreprocessor(BaseEstimator, TransformerMixin):
    """
    A minimal preprocessor that selects the features required by the model.
    It simply converts the specified feature columns to numeric values,
    filling in any missing columns with zeros.
    
    Example usage:
    
        preprocessor = MinimalPreprocessor(model.feature_names_in_)
        X = preprocessor.transform(df)
    """
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Ensure all features exist; if not, add them with zeros.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        # Convert the specified columns to numeric and fill missing values.
        df_processed = df[self.feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        return df_processed.astype(np.float32)

class MinimalClinicalRiskEngine:
    """
    Minimal risk engine to compute uncertainty and anomaly scores.
    
    - Uncertainty is estimated as the entropy (in base 2) of the predicted
      probability distributions for each sample.
    - Anomaly score is computed via an IsolationForest run on RobustScaler-
      transformed data. The raw decision function is normalized between 0 and 1.
    
    Example usage:
    
        engine = MinimalClinicalRiskEngine(model)
        report = engine.analyze(X)
    """
    def __init__(self, model):
        self.model = model
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )
        self.scaler = RobustScaler()
    
    def compute_uncertainty(self, X):
        """
        Computes the uncertainty per sample by applying the entropy function
        to the predicted class probabilities.
        """
        probs = self.model.predict_proba(X)
        uncertainties = np.apply_along_axis(lambda p: entropy(p, base=2), 1, probs)
        return uncertainties

    def compute_anomaly_score(self, X):
        """
        Computes a normalized anomaly score using IsolationForest.
        The scores are normalized between 0 and 1.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.anomaly_detector.fit(X_scaled)
        scores = self.anomaly_detector.decision_function(X_scaled)
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return norm_scores

    def analyze(self, X):
        """
        Runs both uncertainty and anomaly detection.
        
        Returns:
            A pandas DataFrame with columns:
              - 'Uncertainty': The entropy computed per sample.
              - 'Anomaly_Score': The normalized anomaly score per sample.
        """
        uncertainty_scores = self.compute_uncertainty(X)
        anomaly_scores = self.compute_anomaly_score(X)
        return pd.DataFrame({
            'Uncertainty': uncertainty_scores,
            'Anomaly_Score': anomaly_scores
        })

# Usage Example (to be run interactively or in another script):
# ---------------------------------------------------------------
# import joblib, pandas as pd
# from minimal_risk import MinimalPreprocessor, MinimalClinicalRiskEngine
#
# # Load your trained model (ensure your model has a predict_proba method)
# model = joblib.load('path/to/your/model.pkl')
#
# # Read your CSV data into a DataFrame
# df = pd.read_csv('path/to/your/clinical_data.csv')
#
# # Initialize the preprocessor using model feature names
# preprocessor = MinimalPreprocessor(model.feature_names_in_)
# X = preprocessor.transform(df)
#
# # Initialize the risk engine and analyze the data
# engine = MinimalClinicalRiskEngine(model)
# report = engine.analyze(X)
#
# # Save or inspect the report
# report.to_csv('risk_report.csv', index=False)
# ---------------------------------------------------------------
