import logging
import os
import joblib
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from src.models.model_base import ModelBase
from src.utils.display import calculate_and_display_metrics

logger = logging.getLogger(__name__)

class XGBoost(ModelBase):
    model: XGBClassifier

    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.model = XGBClassifier(n_estimators=320, tree_method="hist")
        
    def train(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training XGBoost model...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.debug(f"Fold {fold + 1}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            if fold == 9:
                predicted = self.predict(X_test)
                logger.info("\nXGBoost Metrics:")
                calculate_and_display_metrics(y_test, predicted)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.model is not None:
                joblib.dump(self.model, path)
                print(f"Model successfully saved to {path}")
            else:
                raise ValueError("No model to save. The model hasn't been trained yet.")
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    
    def load(self, path):
        try:
            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model file found at {path}")
            
            # Load the model
            self.model = joblib.load(path)
            print(f"Model successfully loaded from {path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise