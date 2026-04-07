import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    doc = nlp(" ".join(words))
    lemmas = [token.lemma_ for token in doc]

    return " ".join(lemmas)


def create_priority(text):
    text = text.lower()

    if any(word in text for word in ["not working", "failed", "error", "urgent"]):
        return "High"
    elif any(word in text for word in ["delay", "issue", "slow"]):
        return "Medium"
    else:
        return "Low"