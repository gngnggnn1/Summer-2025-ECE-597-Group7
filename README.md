# Phishing Email Detection – Capstone Project

This project aims to design and implement machine learning models to detect phishing emails. The problem is formulated as a binary classification task using traditional ML algorithms trained on manually engineered and TF-IDF features extracted from real phishing email data.

---
## 📁 Folder Structure

- `classifier/`  
  Contains baseline classifiers and all preprocessing code.  
  *(Note: The contents from `Dataset_and_Preprocessing/` and `preprocessing/` should be moved here.)*

- `dataset/`  
  Stores all datasets, including raw emails, cleaned versions, and labeled data.  
  *(Note: Data files from `Dataset_and_Preprocessing/` should also be moved here.)*

- `Report/`  
  Contains the final project report and presentation slides.

- `Dataset_and_Preprocessing/` *(to be merged)*  
  Temporarily holds both data and preprocessing scripts.  
  Will be split and integrated into `Dataset/` and `classifier/`.

- `preprocessing/` *(to be merged)*  
  Contains scripts for feature extraction and data cleaning.  
  Should be moved into `classifier/`.
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

Team members:
Qianjing Sun
Venkata Srivani Reddy Yarram Reddy
Michael He
Junyi Ma
Guangning Li

---

**End of README**
