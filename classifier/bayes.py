import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset.preprocessing import load_data, word_vectorization


def train_bayes():
    tfidf_matrix, keywords, labels = word_vectorization('tfidf')
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", (y_pred == y_test).mean())

train_bayes()
