import pandas as pd

def load_data():
    data = pd.read_csv('../Dataset/phishing_email.csv')
    scam = data[data['label'] == 1]['text_combined'].to_frame(name='text')
    ham = data[data['label'] == 0]['text_combined'].to_frame(name='text')
    return ham, scam

data = load_data()

