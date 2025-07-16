import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import numpy as np
from Dataset.preprocessing import word_vectorization
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from cross_validation import cross_validation
from scipy.sparse import vstack


class SimpleFLD:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "FLD supports only binary classification"

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)

        S1 = np.cov(X1, rowvar=False)
        S2 = np.cov(X2, rowvar=False)
        Sw = S1 + S2

        self.w = np.linalg.pinv(Sw) @ (mean2 - mean1)
        self.threshold = 0.5 * (np.mean(X1 @ self.w) + np.mean(X2 @ self.w))
        return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        proj = X @ self.w
        return (proj > self.threshold).astype(int)

    def predict_proba(self, X):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        proj = X @ self.w
        scaled = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        return np.vstack([1 - scaled, scaled]).T


def evaluate_model(y_true, y_pred, y_proba):
    print("\nClassification Report:\n\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix is:\n\n", confusion_matrix(y_true, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_true, y_proba))
    print("\nAccuracy:", accuracy_score(y_true, y_pred) * 100)


# Load and preprocess data
X_train_vec, X_test_vec, y_train, y_test = word_vectorization(type='tfidf')


# ---------- FLD ----------
print("\nFLD Evaluation: \n")

model = SimpleFLD()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

evaluate_model(y_test, y_pred, y_proba)

# ---------- Cross-Validation FLD ----------
X_all = vstack([X_train_vec, X_test_vec])
y_all = pd.concat([y_train.reset_index(drop=True), y_test.reset_index(drop=True)], ignore_index=True)
results = cross_validation(model, X_all, y_all, cv=5)

print("\nAfter Cross Validation: \n")

for metric, (mean, std) in results.items():
    if metric == 'accuracy':
        print(f"{metric.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    else:
        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
