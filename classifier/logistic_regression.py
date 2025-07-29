# logistic_regression.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from Dataset.preprocessing import word_vectorization

from cross_validation import cross_validation
from scipy.sparse import vstack
import matplotlib.pyplot as plt


def train_and_save_model():
    """
    Trains a Logistic Regression model and saves it to a file.
    """
    print("Loading and vectorizing data...")
    X_train, X_test, y_train, y_test = word_vectorization('tfidf')

    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy*100:.4f}")

def train_with_cross_validation():
    """
    Trains Logistic Regression using cross-validation and prints metrics.
    """
    print("Loading and vectorizing data...")
    X_train, X_test, y_train, y_test = word_vectorization('tfidf')

    X_full = vstack([X_train, X_test])
    y_full = pd.concat([y_train.reset_index(drop=True), y_test.reset_index(drop=True)], ignore_index=True)

    print("Running cross-validation...")
    model = LogisticRegression(max_iter=1000)
    results = cross_validation(model, X_full, y_full, 5)
    
    print("\nCross-Validation Results:")

    for metric, (mean, std) in results.items():
        if metric == 'accuracy':
            print(f"{metric.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
        else:
            print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
    
    k_values = list(range(3, 16)) 
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_means = {m: [] for m in metric_names}
    metric_stds = {m: [] for m in metric_names}

    for k in k_values:
        print(f"\nRunning cross-validation with k={k}...")
        model = LogisticRegression(max_iter=1000)
        results = cross_validation(model, X_full, y_full, cv=k)

        for m in metric_names:
            mean, std = results[m]
            metric_means[m].append(mean)
            metric_stds[m].append(std)

    # Plotting all metrics vs. k
    plt.figure(figsize=(10, 6))
    for m in metric_names:
        plt.errorbar(k_values, metric_means[m], yerr=metric_stds[m],
                     label=m.capitalize(), marker='o', capsize=5)

    plt.title("Effect of Cross-Validation Folds on Logistic Regression Performance")
    plt.xlabel("Number of CV Folds (k)")
    plt.ylabel("Score")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 3: Final model and ROC/Confusion Matrix
    print("\nTraining final Logistic Regression model for ROC and Confusion Matrix...")
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_train, y_train)

    from sklearn.metrics import RocCurveDisplay
    # ROC Curve
    y_proba = final_model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Logistic Regression (on Test Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import ConfusionMatrixDisplay

    # Confusion Matrix
    y_pred = final_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. Logistic Regression without Cross-Validation")
    train_and_save_model()
    print("\n" + "="*60 + "\n")
    print("2. Logistic Regression with Cross-Validation")
    train_with_cross_validation()