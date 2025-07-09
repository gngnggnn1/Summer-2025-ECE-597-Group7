import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    data = pd.read_csv('../dataset/phishing_email.csv')
    scam = data[data['label'] == 1]['text_combined'].to_frame(name='text')
    ham = data[data['label'] == 0]['text_combined'].to_frame(name='text')
    return ham, scam

def word_vectorization(type):
    ham, scam = load_data()
    ham = ham.dropna(subset=["text"])
    scam = scam.dropna(subset=["text"])
    if type == 'tfidf':
        all_text = pd.concat([ham["text"], scam["text"]], ignore_index=True)
        # here we removed stopwords.
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_text)

        keywords = vectorizer.get_feature_names_out()
        ham_labels = pd.Series([0] * len(ham), name='label')
        spam_labels = pd.Series([1] * len(scam), name='label')
        labels = pd.concat([ham_labels, spam_labels], ignore_index=True)
        return tfidf_matrix, keywords, labels