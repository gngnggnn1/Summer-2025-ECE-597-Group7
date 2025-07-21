from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def cross_validation(model, X, y, cv=5):

    """
    Run cross-validation on a given model and return accuracy, precision, recall, and F1-score.

    Parameters:
    - model: scikit-learn estimator
    - X: feature matrix (sparse or dense)
    - y: labels
    - cv: number of folds (default 5)

    Returns:
    - dict: average scores and std dev for each metric
    """

    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    
    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred))
        rec_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))

    return {
        'accuracy': (np.mean(acc_scores), np.std(acc_scores)),
        'precision': (np.mean(prec_scores), np.std(prec_scores)),
        'recall': (np.mean(rec_scores), np.std(rec_scores)),
        'f1': (np.mean(f1_scores), np.std(f1_scores))
    }
