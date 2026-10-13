import joblib
from pathlib import Path


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# LOAD ML MODEL
# --------------------------------------------------

topic_classifier = joblib.load(
    MODEL_DIR / "topic_classifier.pkl"
)

tfidf_vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# TOPIC PREDICTION
# --------------------------------------------------

def predict_topic(text):

    text_vector = tfidf_vectorizer.transform([text])

    prediction = topic_classifier.predict(text_vector)

    return prediction[0]


# --------------------------------------------------
# PREDICTION WITH CONFIDENCE
# --------------------------------------------------

def predict_topic_with_confidence(text):

    text_vector = tfidf_vectorizer.transform([text])

    prediction = topic_classifier.predict(text_vector)[0]

    confidence = None

    if hasattr(topic_classifier, "predict_proba"):

        probabilities = topic_classifier.predict_proba(text_vector)[0]

        confidence = probabilities.max()

    return prediction, confidence