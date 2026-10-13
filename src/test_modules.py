from ml_model import predict_topic_with_confidence
from ese_analysis import analyze_abstract


text = """
This study proposes a machine learning framework for predicting student
academic performance using demographic, attendance, and assessment data.
Several supervised learning algorithms were evaluated using a dataset
collected from undergraduate students. The Random Forest model achieved
an accuracy of 91%, outperforming logistic regression and decision tree
models. The findings indicate that attendance and previous assessment
scores are important predictors of academic performance. The proposed
approach may support early identification of students who require
additional academic assistance.
"""


# ------------------------------
# ML
# ------------------------------

topic, confidence = predict_topic_with_confidence(text)

print("PREDICTED TOPIC:", topic)

if confidence is not None:
    print("CONFIDENCE:", round(confidence * 100, 2), "%")


# ------------------------------
# ESE
# ------------------------------

result = analyze_abstract(text)

print("\nDOCUMENT LEVEL")
print(result["Document Level"])

print("\nSENTENCE LEVEL")

for sentence in result["Sentence Level"]:
    print(
        sentence["Sentence No"],
        sentence["Role"],
        sentence["Sentence"]
    )

print("\nASPECT LEVEL")

for aspect in result["Aspect Level"]:
    print(aspect)