from enum import Enum
from datetime import datetime, time
import logging
from typing import Dict, Optional
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

def calculate_and_display_metrics(y_true, y_pred):
    precision_real = precision_score(y_true, y_pred, pos_label=1)
    recall_real = recall_score(y_true, y_pred, pos_label=1)
    f1_real = f1_score(y_true, y_pred, pos_label=1)
    mcc_real = matthews_corrcoef(y_true, y_pred)
    roc_real = roc_auc_score(y_true, y_pred)

    precision_fake = precision_score(y_true, y_pred, pos_label=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=0)
    mcc_fake = mcc_real 
    roc_fake = roc_real

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    mcc_weighted = mcc_real 
    roc_weighted = roc_real

    metrics = {
        "Class": ["Real", "Fake", "Weighted Average"],
        "Precision": [precision_real, precision_fake, precision_weighted],
        "Recall": [recall_real, recall_fake, recall_weighted],
        "F1-Score": [f1_real, f1_fake, f1_weighted],
        "MCC": [mcc_real, mcc_fake, mcc_weighted],
        "ROC": [roc_real, roc_fake, roc_weighted]
    }

    df = pd.DataFrame(metrics)

    logger.info("\n----------------------\n" + df.to_string(index=False) + "\n----------------------\n")

def display_predictions_timeline(predictions: np.ndarray, probabilities: np.ndarray) -> None:

    seconds_per_row = 30
    total_predictions = len(predictions)
    
    for row_start in range(0, total_predictions, seconds_per_row):
        row_end = min(row_start + seconds_per_row, total_predictions)
        
        print("\nTime     ", end="")
        for i in range(row_start, row_end):
            minutes = i // 60
            seconds = i % 60
            print(f"{minutes:02d}:{seconds:02d}  ", end="")
            
        # Predictions row
        print("\nPred     ", end="")
        for i in range(row_start, row_end):
            print(f"{predictions[i]:^5}  ", end="")
            
        # Probabilities row
        print("\nProb     ", end="")
        for i in range(row_start, row_end):
            print(f"{max(probabilities[i]):5.3f}", end="  ")
        print("\n")