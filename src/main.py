import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

import xgboost as xgb

def calculate_and_display_metrics(y_true, y_pred):
    # Calculate per-class metrics
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

    # Weighted Average (using precision_recall_fscore_support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    mcc_weighted = mcc_real 
    roc_weighted = roc_real

    # Create a dictionary of all metrics
    metrics = {
        "Class": ["Real", "Fake", "Weighted Average"],
        "Precision": [precision_real, precision_fake, precision_weighted],
        "Recall": [recall_real, recall_fake, recall_weighted],
        "F1-Score": [f1_real, f1_fake, f1_weighted],
        "MCC": [mcc_real, mcc_fake, mcc_weighted],
        "ROC": [roc_real, roc_fake, roc_weighted]
    }

    # Create a pandas DataFrame to display the metrics in a nice table
    df = pd.DataFrame(metrics)

    # Display the table
    print(df.to_string(index=False))


data_path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")

print("Path to dataset files:", data_path)

dtype = [('chroma_stft', 'float'), 
        ('rms', 'float'),
        ('spectral_centroid', 'float'),
        ('spectral_bandwidth', 'float'),
        ('rolloff', 'float'),
        ('zero_crossing_rate', 'float'),
        ('mfcc1', 'float'),
        ('mfcc2', 'float'),
        ('mfcc3', 'float'),
        ('mfcc4', 'float'),
        ('mfcc5', 'float'),
        ('mfcc6', 'float'),
        ('mfcc7', 'float'),
        ('mfcc8', 'float'),
        ('mfcc9', 'float'),
        ('mfcc10', 'float'),
        ('mfcc11', 'float'),
        ('mfcc12', 'float'),
        ('mfcc13', 'float'),
        ('mfcc14', 'float'),
        ('mfcc15', 'float'),
        ('mfcc16', 'float'),
        ('mfcc17', 'float'),
        ('mfcc18', 'float'),
        ('mfcc19', 'float'),
        ('mfcc20', 'float'),
        ('label', 'S10')]

pathToCsv = os.path.join(data_path, "KAGGLE/DATASET-balanced.csv")

data = np.loadtxt(pathToCsv, delimiter=',', dtype=dtype, skiprows=1) 

print(data[0])
X = np.array([(row[0], row[1]) for row in data])
y = np.array([row[-1].decode('utf-8') for row in data])

y = np.where(y == "FAKE", 0, 1).astype(int)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = {}

model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)


for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predicted = model.predict(X_test)

    calculate_and_display_metrics(y_test, predicted)
    print("---------------------------------------------")

model.save_model("clf.json")