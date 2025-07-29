import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import vstack

from Dataset.preprocessing import word_vectorization
from classifier.cross_validation import cross_validation
from sklearn.metrics import RocCurveDisplay


def train_and_save_naive_bayes():
    """
    Trains a Naive Bayes model without cross-validation and prints evaluation metrics.
    """
    print("Loading and vectorizing data...")
    X_train, X_test, y_train, y_test = word_vectorization('tfidf')

    print("Training the Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Training complete.")

    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy * 100:.4f}")


def train_naive_bayes_with_cross_validation():
    """
    Trains Naive Bayes using cross-validation and prints results for k=5.
    Also plots how performance varies across different k values.
    """
    print("Loading and vectorizing data...")
    X_train, X_test, y_train, y_test = word_vectorization('tfidf')

    X_full = vstack([X_train, X_test])
    y_full = pd.concat([y_train.reset_index(drop=True), y_test.reset_index(drop=True)], ignore_index=True)

    # Step 1: Print results for k = 5
    print("\nRunning cross-validation with k=5...")
    model = MultinomialNB()
    results_k5 = cross_validation(model, X_full, y_full, cv=5)

    print("\nCross-Validation Results (k=5):")
    for metric, (mean, std) in results_k5.items():
        if metric == 'accuracy':
            print(f"{metric.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
        else:
            print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")

    # Step 2: Plot metrics across multiple k-values
    k_values = list(range(3, 16))
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_means = {m: [] for m in metric_names}
    metric_stds = {m: [] for m in metric_names}

    for k in k_values:
        print(f"\nRunning cross-validation with k={k}...")
        model = MultinomialNB()
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

    plt.title("Effect of Cross-Validation Folds on Naive Bayes Performance")
    plt.xlabel("Number of CV Folds (k)")
    plt.ylabel("Score")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Step 3: Plot ROC Curve using held-out test set
    print("\nTraining final model for ROC curve visualization...")
    final_model = MultinomialNB()
    final_model.fit(X_train, y_train)
    y_proba = final_model.predict_proba(X_test)[:, 1]  # Probability of positive class

    print("Plotting ROC Curve...")
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Naive Bayes (on Test Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import ConfusionMatrixDisplay

    y_pred = final_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Naive Bayes")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("1. Naive Bayes without Cross-Validation")
    train_and_save_naive_bayes()
    print("\n" + "=" * 60 + "\n")
    print("2. Naive Bayes with Cross-Validation and Plotting")
    train_naive_bayes_with_cross_validation()



