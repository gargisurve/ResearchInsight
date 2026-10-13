import pickle
import re
import nltk

from pathlib import Path
from nltk.tokenize import sent_tokenize


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# NLTK RESOURCES
# --------------------------------------------------

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# --------------------------------------------------
# LOAD ESE CONFIGURATION
# --------------------------------------------------

with open(MODEL_DIR / "ese_config.pkl", "rb") as file:
    ese_config = pickle.load(file)


role_keywords = ese_config["role_keywords"]
aspect_keywords = ese_config["aspect_keywords"]
stance_patterns = ese_config["stance_patterns"]

# --------------------------------------------------
# SENTENCE-LEVEL ANALYSIS
# --------------------------------------------------

def get_role_scores(sentence):

    text = sentence.lower()

    scores = {
        "Background": 0,
        "Objective": 0,
        "Method": 0,
        "Finding": 0,
        "Comparison": 0,
        "Conclusion": 0
    }

    for role, keywords in role_keywords.items():

        for keyword in keywords:

            if keyword in text:
                scores[role] += 1

    # Numerical results
    if re.search(r"\b\d+(\.\d+)?\s*%", text):
        scores["Finding"] += 2

    # Comparison indicators
    if re.search(
        r"\b(better|worse|higher|lower|more|less|outperforms)\b",
        text
    ):
        scores["Comparison"] += 1

    return scores


def classify_sentence_role(sentence):

    text = sentence.lower()

    scores = get_role_scores(sentence)

    # -----------------------------------------
    # Strong conclusion indicators
    # -----------------------------------------

    conclusion_phrases = [
        "in conclusion",
        "we conclude",
        "we conclude that",
        "our findings suggest",
        "our findings indicate",
        "these findings suggest",
        "these findings indicate",
        "the findings suggest",
        "the findings indicate",
        "may support",
        "may help",
        "could support",
        "could help",
        "can support",
        "can help",
        "has potential",
        "have potential",
        "potential to",
        "implications"
    ]

    if any(phrase in text for phrase in conclusion_phrases):
        return "Conclusion"

    # -----------------------------------------
    # Strong finding indicators
    # -----------------------------------------

    finding_phrases = [
        "results show",
        "results demonstrate",
        "results indicate",
        "results suggest",
        "findings show",
        "findings demonstrate",
        "findings indicate",
        "findings suggest",
        "we found",
        "found that",
        "achieved",
        "outperformed",
        "demonstrated"
    ]

    if any(phrase in text for phrase in finding_phrases):
        return "Finding"

    # -----------------------------------------
    # Explicit comparison
    # -----------------------------------------

    comparison_phrases = [
        "better than",
        "worse than",
        "higher than",
        "lower than",
        "outperforms",
        "outperformed",
        "superior to",
        "inferior to",
        "compared with",
        "compared to"
    ]

    if any(phrase in text for phrase in comparison_phrases):
        return "Comparison"

    # -----------------------------------------
    # Objective
    # -----------------------------------------

    objective_phrases = [
        "aims to",
        "aimed to",
        "objective is",
        "goal is",
        "purpose is",
        "we propose",
        "we present",
        "we develop",
        "we introduce",
        "we investigate",
        "we examine",
        "we explore",
        "this study proposes",
        "this study aims",
        "this paper proposes"
    ]

    if any(phrase in text for phrase in objective_phrases):
        return "Objective"

    # -----------------------------------------
    # Method
    # -----------------------------------------

    method_phrases = [
        "we used",
        "we use",
        "we applied",
        "we apply",
        "we employed",
        "we conducted",
        "was applied",
        "were applied",
        "was used",
        "were used",
        "was evaluated",
        "were evaluated",
        "using a dataset",
        "using the dataset",
        "trained using",
        "trained on",
        "experiments were"
    ]

    if any(phrase in text for phrase in method_phrases):
        return "Method"

    # -----------------------------------------
    # Numerical result
    # -----------------------------------------

    if re.search(r"\b\d+(\.\d+)?\s*%", text):
        return "Finding"

    # -----------------------------------------
    # Existing scoring as fallback
    # -----------------------------------------

    max_score = max(scores.values())

    if max_score == 0:
        return "Background"

    return max(scores, key=scores.get)


def analyze_sentences(text):

    sentences = sent_tokenize(text)

    sentence_analysis = []

    for i, sentence in enumerate(sentences, 1):

        role = classify_sentence_role(sentence)

        sentence_analysis.append({
            "Sentence No": i,
            "Sentence": sentence,
            "Role": role
        })

    return sentence_analysis


# --------------------------------------------------
# ASPECT-LEVEL ANALYSIS
# --------------------------------------------------

def detect_aspects(sentence):

    text = sentence.lower()

    detected = []

    # Model / Method
    method_terms = [
        "algorithm",
        "model",
        "method",
        "framework",
        "technique",
        "classifier",
        "regression",
        "neural network",
        "machine learning",
        "deep learning"
    ]

    if any(term in text for term in method_terms):
        detected.append("Model / Method")

    # Performance
    performance_terms = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1-score",
        "auc",
        "roc",
        "performance",
        "prediction accuracy",
        "error rate"
    ]

    if any(term in text for term in performance_terms):
        detected.append("Performance")

    # Dataset / Data
    data_terms = [
        "dataset",
        "data were collected",
        "data was collected",
        "participants",
        "observations",
        "sample size",
        "variables",
        "features"
    ]

    if any(term in text for term in data_terms):
        detected.append("Dataset / Data")

    # Efficiency / Cost
    efficiency_terms = [
        "computational cost",
        "computational efficiency",
        "runtime",
        "resource usage",
        "memory usage",
        "processing time"
    ]

    if any(term in text for term in efficiency_terms):
        detected.append("Efficiency / Cost")

    # Application
    application_terms = [
        "application",
        "applications",
        "healthcare",
        "medicine",
        "education",
        "finance",
        "agriculture",
        "clinical",
        "diagnosis",
        "early identification"
    ]

    if any(term in text for term in application_terms):
        detected.append("Application")

    # Limitations
    limitation_terms = [
        "limitation",
        "limitations",
        "challenge",
        "challenges",
        "constraint",
        "constraints",
        "drawback",
        "risk"
    ]

    if any(term in text for term in limitation_terms):
        detected.append("Limitations / Challenges")

    # Relationships
    relationship_terms = [
        "association",
        "associated with",
        "correlation",
        "correlated with",
        "relationship between",
        "related to",
        "linked to"
    ]

    if any(term in text for term in relationship_terms):
        detected.append("Relationships / Associations")

    return detected


# --------------------------------------------------
# RESEARCH STANCE
# --------------------------------------------------

def detect_stance(sentence):

    text = sentence.lower()

    detected_stances = []

    for stance, keywords in stance_patterns.items():

        for keyword in keywords:

            if keyword in text:
                detected_stances.append(stance)
                break

    if len(detected_stances) == 0:
        return "Research finding reported"

    return ", ".join(detected_stances)


def analyze_aspects(text):

    sentences = sent_tokenize(text)

    aspect_results = []

    for sentence in sentences:

        aspects = detect_aspects(sentence)

        if len(aspects) > 0:

            stance = detect_stance(sentence)

            for aspect in aspects:

                aspect_results.append({
                    "Aspect": aspect,
                    "Evidence": sentence,
                    "Research Stance": stance
                })

    return aspect_results


# --------------------------------------------------
# DOCUMENT-LEVEL ANALYSIS
# --------------------------------------------------

def document_level_analysis(text):

    sentence_results = analyze_sentences(text)

    objective = []
    methods = []
    findings = []
    conclusions = []

    for item in sentence_results:

        role = item["Role"]
        sentence = item["Sentence"]

        if role == "Objective":
            objective.append(sentence)

        elif role == "Method":
            methods.append(sentence)

        elif role in ["Finding", "Comparison"]:
            findings.append(sentence)

        elif role == "Conclusion":
            conclusions.append(sentence)

    return {
        "Objective": objective,
        "Method": methods,
        "Findings": findings,
        "Conclusion": conclusions
    }


# --------------------------------------------------
# COMPLETE ESE ANALYSIS
# --------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def analyze_abstract(text):

    text = clean_text(text)

    document_analysis = document_level_analysis(text)

    sentence_analysis = analyze_sentences(text)

    aspect_analysis = analyze_aspects(text)

    return {
        "Document Level": document_analysis,
        "Sentence Level": sentence_analysis,
        "Aspect Level": aspect_analysis
    }