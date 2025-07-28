import numpy as np
from cross_validation import cross_validation
from dataset.preprocessing import word_vectorization
from sklearn.metrics import classification_report

# Load data
X_train_vec, X_test_vec, y_train, y_test = word_vectorization('tfidf')
X_all = np.vstack([X_train_vec.toarray(), X_test_vec.toarray()])
y_all = np.concatenate([y_train.values, y_test.values])



# ---------- Linear Regression ----------

from sklearn.linear_model import LogisticRegression

print("\nLinear Regression Evaluation: \n")

class SimpleLinearRegression:
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        return self.model.predict(X)

    def predict_proba(self, X):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        return self.model.predict_proba(X)

lin_model = SimpleLinearRegression()
lin_model.fit(X_train_vec, y_train)
y_pred = lin_model.predict(X_test_vec)
y_proba = lin_model.predict_proba(X_test_vec)[:, 1]


# ---------- Cross-Validation Linear Regression ----------

results_lin = cross_validation(lin_model, X_all, y_all, cv=5)

print("\nAfter Cross Validation (Linear Regression): \n")

for metric, (mean, std) in results_lin.items():
    if metric == 'accuracy':
        print(f"{metric.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    else:
        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")

print(classification_report(y_test, y_pred))
print("Accuracy On Test Set:", (y_pred == y_test).mean())