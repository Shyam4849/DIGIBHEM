
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = text.split()
    return ' '.join([stemmer.stem(word) for word in tokens if word not in stop_words])

def load_and_prepare_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true])[["title", "label"]]
    df = df.sample(frac=1).reset_index(drop=True)
    df["clean_text"] = df["title"].apply(preprocess_text)
    df.to_csv("processed_news.csv", index=False)
    return df

if __name__ == "__main__":
    load_and_prepare_data()
