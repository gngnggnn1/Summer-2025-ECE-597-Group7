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