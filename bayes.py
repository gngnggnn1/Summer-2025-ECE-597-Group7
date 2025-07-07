import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    data = pd.read_csv('../preprocessing/ready_for_training.csv')

    scam = data[data['label'] == 1]['processed_text'].to_frame(name='text')
    ham = data[data['label'] == 0]['processed_text'].to_frame(name='text')
    return ham, scam


def key_world_extraction(type):
    ham, scam = load_data()
    ham = ham.dropna(subset=["text"])
    scam = scam.dropna(subset=["text"])
    if type == 'tfidf':
        all_text = pd.concat([ham["text"], scam["text"]], ignore_index=True)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_text)

        keywords = vectorizer.get_feature_names_out()
        ham_labels = pd.Series([0] * len(ham), name='label')
        spam_labels = pd.Series([1] * len(scam), name='label')
        labels = pd.concat([ham_labels, spam_labels], ignore_index=True)
        return tfidf_matrix, keywords, labels

def train_bayes():
    tfidf_matrix, keywords, labels = key_world_extraction('tfidf')
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

train_bayes()
