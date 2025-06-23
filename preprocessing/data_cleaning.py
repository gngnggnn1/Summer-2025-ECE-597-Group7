from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import email
from email import policy

def extract_subject_body(raw_text):
    msg = email.message_from_string(raw_text, policy=policy.default)
    subject = msg["Subject"] or ""
    # 获取纯文本正文
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
        else:
            body = ""
    else:
        body = msg.get_content()
    return subject, body

def clean_data():
    spam = pd.read_csv('./dataset/CaptstoneProjectData_2025.csv')
    spam = spam.iloc[:, :2]
    spam = spam.dropna()

    df = pd.read_csv('./dataset/emails.csv')

    # 应用抽取
    subjects = []
    bodies = []
    for msg in df["message"]:
        subject, body = extract_subject_body(msg)
        subjects.append(subject)
        bodies.append(body)
    df["Subject"] = subjects
    df["Body"] = bodies
    ham = pd.DataFrame({"Subject": subjects, "Body": bodies})
    ham = ham.dropna()

    spam.to_csv('./dataset/spam_cleaned.csv', index=False)
    ham.to_csv('./dataset/ham_cleaned.csv', index=False)

def load_data():
    ham = pd.read_csv('./dataset/ham_cleaned.csv')
    spam = pd.read_csv('./dataset/spam_cleaned.csv')
    return ham, spam

def key_world_extraction(type):
    ham, spam = load_data()
    if type == 'tfidf':
        # 合并 subject 和 body 为 text 字段
        ham["text"] = ham["Subject"].fillna('') + " " + ham["Body"].fillna('')
        spam["text"] = spam["Subject"].fillna('') + " " + spam["Body"].fillna('')

        all_text = pd.concat([ham["text"], spam["text"]], ignore_index=True)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_text)

        keywords = vectorizer.get_feature_names_out()
        # 返回 tfidf_matrix, keywords, labels
        # 生成标签：假设 ham 为 0，spam 为 1
        ham_labels = pd.Series([0] * len(ham), name='label')
        spam_labels = pd.Series([1] * len(spam), name='label')
        labels = pd.concat([ham_labels, spam_labels], ignore_index=True)
        return tfidf_matrix, keywords, labels


# 朴素贝叶斯训练函数
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_bayes():
    tfidf_matrix, keywords, labels = key_world_extraction('tfidf')
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
