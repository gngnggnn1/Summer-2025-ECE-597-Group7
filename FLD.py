import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score


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
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))
    print("Accuracy:", accuracy_score(y_true, y_pred) * 100)


# Load and preprocess data
data = pd.read_csv('preprocessing/ready_for_training.csv')
data = data.dropna(subset=['processed_text'])

X_text = data['processed_text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(X_text)

# ---------- FLD ----------
print("===== Normal FLD Evaluation =====")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

model = SimpleFLD()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred, y_proba)

# ---------- Cross-Validation FLD ----------
print("\n===== Cross-Validation FLD Evaluation =====")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = []
all_probas = []
all_trues = []

for train_idx, test_idx in kf.split(X_tfidf, y):
    X_train_cv, X_test_cv = X_tfidf[train_idx], X_tfidf[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

    model_cv = SimpleFLD()
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)
    y_proba_cv = model_cv.predict_proba(X_test_cv)[:, 1]

    all_preds.extend(y_pred_cv)
    all_probas.extend(y_proba_cv)
    all_trues.extend(y_test_cv)

evaluate_model(all_trues, all_preds, all_probas)