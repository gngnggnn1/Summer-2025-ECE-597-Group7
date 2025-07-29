import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

'''
Instructions:
just simply call the function word_vectorization()
and the type could be:
'tfidf', 'bow', ...(we may add something new)
'''

def load_data():
    data = pd.read_csv('../data/phishing_email.csv')
    scam = data[data['label'] == 1]['text_combined'].to_frame(name='text')
    ham = data[data['label'] == 0]['text_combined'].to_frame(name='text')
    return ham, scam

def word_vectorization(type='tfidf'):
    ham, scam = load_data()
    ham = ham.dropna(subset=["text"])
    scam = scam.dropna(subset=["text"])
    all_text = pd.concat([ham["text"], scam["text"]], ignore_index=True)
    ham_labels = pd.Series([0] * len(ham), name='label')
    spam_labels = pd.Series([1] * len(scam), name='label')
    labels = pd.concat([ham_labels, spam_labels], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(all_text, labels, test_size=0.2)

    if type == 'tfidf':
        # here we removed stopwords.
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    elif type == 'bow':
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, use_idf=False, norm=None)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test



