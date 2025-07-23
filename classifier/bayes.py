import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset.preprocessing import load_data, word_vectorization

from classifier.cross_validation import cross_validation


def train_bayes(cv = True):
    X_train, X_test, y_train, y_test = word_vectorization()
    model = MultinomialNB()
    if cv:
        result = cross_validation(model,X_train,y_train)

        for metric, (mean, std) in result.items():
            print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy On Test Set:", (y_pred == y_test).mean())

train_bayes()


