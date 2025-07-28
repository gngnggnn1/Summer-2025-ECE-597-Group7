# logistic_regression.py

import pandas as pd
import joblib  # <-- Import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset.preprocessing import word_vectorization


def train_and_save_model():
    """
    Trains a Logistic Regression model and saves it to a file.
    """
    print("Loading and vectorizing data...")
    tfidf_matrix, _, labels = word_vectorization('bag')

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    print("Training the Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- Step to export the model ---
    model_filename = 'logistic_regression_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModel has been saved to '{model_filename}'")
    # ---------------------------------

    # Evaluate the model (optional, but good practice)
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_and_save_model()