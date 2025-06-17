# Phishing Email Detection – Capstone Project

This project aims to design and implement machine learning models to detect phishing emails. The problem is formulated as a binary classification task using traditional ML algorithms trained on manually engineered and TF-IDF features extracted from real phishing email data.

---

## Project Stages

| Stage                                  | Status      | Description                                                                                                                                                               |
| -------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Literature Review and Learning      | Completed   | Studied prior research on phishing detection and email analysis techniques. Discussed challenges and strategies in detecting abnormalities through email characteristics. |
| 2. Data Understanding                  | Completed   | Explored email dataset structure, including subject, sender info, body text, and typical features.                                                                        |
| 3. Preprocessing & Feature Engineering | In Progress | Cleaning emails, tokenization, stop words removal, HTML handling, and extraction of structural and TF-IDF features.                                                       |
| 4. Machine Learning Model Training     | In Progress | Training algorithms including Naive Bayes, Logistic Regression, Random Forest, FLD, and Nearest Neighbor.                                                                 |
| 5. Evaluation & Metrics                | Upcoming    | Evaluate models using AUC, confusion matrix, and balanced accuracy metrics.                                                                                               |
| 6. Optimization & Validation           | Upcoming    | Address class imbalance, apply cross-validation, and test on hold-out dataset.                                                                                            |
| 7. Semantic Embeddings                 | Upcoming    | Replace TF-IDF with dense semantic embeddings for advanced feature representation.                                                                                        |
| 8. Reporting & CLI Tool                | Upcoming    | Summarize findings and optionally deploy the model as a script or CLI tool.                                                                                               |

---

## Project Structure

```

phishing-email-detector/
├── data/ # Raw input data
│ └── CapstoneProjectData_2025.csv # Raw phishing emails dataset
├── features/ # Engineered features and labels
│ ├── features.csv # Extracted feature matrix
│ └── labels.csv # Labels (currently all 1s, to update with real labels)
├── models/ # Saved ML models (to be added)
├── src/ # Core Python scripts
│ ├── phishing_email_preprocessing.py # Feature extraction and preprocessing
│ └── train_model.py # ML model training (to be added)
├── notebooks/ # Exploratory data analysis and experiments (optional)
├── requirements.txt # Python dependencies
└── README.md # This file

```

---

## Features Extracted

### Structural (manual features)

- Character count
- Word count
- Uppercase ratio
- Number of URLs
- Number of exclamation marks

### Textual (semantic features)

- Top 300 most informative words identified via TF-IDF on cleaned email content

---

## Team Tasks and Progress

Each team member is responsible for implementing and evaluating one classification algorithm. The final optimization and validation will be done together.

| Team Member                        | Assigned Algorithm                 | Status      |
| ---------------------------------- | ---------------------------------- | ----------- |
| Michael He                         | Naive Bayes                        | In Progress |
| Guangning Li                       | Logistic Regression                | In Progress |
| Junyi Ma                           | Random Forest                      | In Progress |
| Venkata Srivani Reddy Yarram Reddy | Fisher’s Linear Discriminant (FLD) | In Progress |
| Qianjing Sun                       | Nearest Neighbor                   | In Progress |
| All                                | Model Optimization & Validation    | Upcoming    |

---

# Contact

Instructor: Riham AlTawy
GitHub: [https://github.com/RihamAlTawy](https://github.com/RihamAlTawy)

---

**End of README**
