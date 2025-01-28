import logging
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from src.models.model_base import ModelBase
from src.utils.display import calculate_and_display_metrics

logger = logging.getLogger(__name__)

class RandomForest(ModelBase):
    model: RandomForestClassifier

    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.model = RandomForestClassifier(n_estimators=310, random_state=42)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training RandomForest model...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.debug(f"Fold {fold + 1}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)

            if fold == 9:
                predicted = self.predict(X_test)
                logger.info("\nRandomForest Metrics:")
                calculate_and_display_metrics(y_test, predicted)
    
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.model is not None:
                joblib.dump(self.model, path)
                logger.info(f"Model successfully saved to {path}")
            else:
                raise ValueError("No model to save. The model hasn't been trained yet.")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model file found at {path}")
            
            self.model = joblib.load(path)
            logger.info(f"Model successfully loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

