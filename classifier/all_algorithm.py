import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset.preprocessing import load_data, word_vectorization

from classifier.cross_validation import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_bayes(cv = True):
    X_train, X_test, y_train, y_test = word_vectorization()
    model = MultinomialNB()
    if cv:
        result = cross_validation(model, X_train, y_train)
        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nBayes Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())


def train_logistic(cv=True):
    """Train and evaluate a Logistic Regression classifier."""
    X_train, X_test, y_train, y_test = word_vectorization()
    model = LogisticRegression(max_iter=1000)
    if cv:
        result = cross_validation(model, X_train, y_train)
        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nLogistic Regression Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())


def train_fld(cv=True):
    """Train and evaluate a Fisher Linear Discriminant (LDA) classifier."""
    X_train, X_test, y_train, y_test = word_vectorization()
    model = LinearDiscriminantAnalysis()
    if cv:
        # Convert sparse to dense for LDA cross-validation
        X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
        result = cross_validation(model, X_train_dense, y_train)
        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
    model.fit(X_train.toarray() if hasattr(X_train, 'toarray') else X_train, y_train)
    X_test_arr = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    y_pred = model.predict(X_test_arr)
    print("\nFisher LDA Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())


def train_random_forest(cv=True):
    """Train and evaluate a Random Forest classifier."""
    X_train, X_test, y_train, y_test = word_vectorization()
    model = RandomForestClassifier(n_estimators=100)
    if cv:
        result = cross_validation(model, X_train, y_train)
        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nRandom Forest Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())


def train_knn(cv=True):
    """Train and evaluate a K-Nearest Neighbors classifier."""
    X_train, X_test, y_train, y_test = word_vectorization()
    model = KNeighborsClassifier(n_neighbors=5)
    if cv:
        result = cross_validation(model, X_train, y_train)
        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nKNN Results:")
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())

# Example calls:
train_bayes()
train_logistic()
train_fld()
train_random_forest()
train_knn()
