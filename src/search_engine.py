import os
import pickle
import re
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD SEARCH ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "search_engine.pkl"
)


with open(
    MODEL_PATH,
    "rb"
) as file:

    search_engine = pickle.load(file)


# ============================================================
# EXTRACT COMPONENTS
# ============================================================

search_vectorizer = (
    search_engine["vectorizer"]
)

paper_matrix = (
    search_engine["paper_matrix"]
)

paper_embeddings = (
    search_engine["paper_embeddings"]
)

topic_embeddings = (
    search_engine["topic_embeddings"]
)

papers = (
    search_engine["papers"]
)

topics = (
    search_engine["topics"]
)


print(
    "Search engine loaded successfully."
)

print(
    "Papers:",
    len(papers)
)

print(
    "Topics:",
    len(topics)
)


# ============================================================
# LOAD SEMANTIC MODEL ONCE
# ============================================================

from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_semantic_model():

    print("Loading semantic model...")

    model = SentenceTransformer(
        search_engine["semantic_model_name"]
    )

    print(
        "Semantic model loaded successfully."
    )

    return model


semantic_model = get_semantic_model()


# ============================================================
# TEXT CLEANING
# ============================================================

import re


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# QUERY EMBEDDING
# ============================================================

def get_query_embedding(query):

    return semantic_model.encode(
        [query],
        normalize_embeddings=True
    )[0]


# ============================================================
# RELATED TOPICS
# ============================================================

def detect_related_topics(
    query,
    top_n=5
):

    query_clean = clean_text(
        query
    )

    query_embedding = (
        get_query_embedding(
            query_clean
        )
    )

    topic_scores = []

    for topic, profile in (
        topic_embeddings.items()
    ):

        score = np.dot(
            query_embedding,
            profile
        )

        topic_scores.append(
            (
                topic,
                float(score)
            )
        )

    topic_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return topic_scores[:top_n]


# ============================================================
# SEARCH PAPERS
# ============================================================

def search_papers(
    query,
    top_k=5
):

    query_clean = clean_text(
        query
    )


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not query_clean:

        return {
            "query": query,
            "related_topics": [],
            "results": [],
            "message": (
                "Please enter a research query."
            )
        }


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_embedding = (
        get_query_embedding(
            query_clean
        )
    )


    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    semantic_scores = np.dot(
        paper_embeddings,
        query_embedding
    )


    # ========================================================
    # TF-IDF SIMILARITY
    # ========================================================

    query_vector = (
        search_vectorizer.transform(
            [query_clean]
        )
    )


    from sklearn.metrics.pairwise import (
        cosine_similarity
    )


    tfidf_scores = (
        cosine_similarity(
            query_vector,
            paper_matrix
        ).flatten()
    )


    # ========================================================
    # RELATED TOPICS
    # ========================================================

    related_topics = (
        detect_related_topics(
            query,
            top_n=5
        )
    )


    # ========================================================
    # TOPIC SCORE FOR EACH PAPER
    # ========================================================

    paper_topic_scores = np.zeros(
        len(papers)
    )


    for topic, score in (
        related_topics
    ):

        topic_mask = (
            papers["Topic"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            topic.lower()
        )


        paper_topic_scores[
            topic_mask.values
        ] = score


    # ========================================================
    # NORMALIZE TF-IDF
    # ========================================================

    if tfidf_scores.max() > 0:

        tfidf_normalized = (
            tfidf_scores /
            tfidf_scores.max()
        )

    else:

        tfidf_normalized = (
            tfidf_scores
        )


    # ========================================================
    # NORMALIZE SEMANTIC SCORE
    # ========================================================

    semantic_min = (
        semantic_scores.min()
    )

    semantic_max = (
        semantic_scores.max()
    )


    if semantic_max > semantic_min:

        semantic_normalized = (
            semantic_scores -
            semantic_min
        ) / (
            semantic_max -
            semantic_min
        )

    else:

        semantic_normalized = (
            semantic_scores
        )


    # ========================================================
    # FINAL RANKING
    # ========================================================

    ranking_scores = (

        0.55 *
        semantic_normalized

        +

        0.30 *
        tfidf_normalized

        +

        0.15 *
        paper_topic_scores
    )


    # ========================================================
    # RELEVANCE CHECK
    # ========================================================

    best_semantic = (
        semantic_scores.max()
    )

    best_tfidf = (
        tfidf_scores.max()
    )


    query_words = set(
        query_clean.split()
    )


    vocabulary = set(
        search_vectorizer
        .vocabulary_
        .keys()
    )


    known_words = (
        query_words
        & vocabulary
    )


    generic_words = {

        "paper",
        "research",
        "study",
        "analysis",
        "method",
        "methods",
        "model",
        "models",
        "data",
        "using",
        "based",
        "system",
        "systems",
        "approach",
        "prediction",
        "predict",
        "results",
        "performance"
    }


    meaningful_words = (
        known_words
        - generic_words
    )


    has_meaningful_keyword = (
        len(meaningful_words) >= 1
    )


    has_strong_tfidf = (
        best_tfidf >= 0.08
    )


    has_strong_semantic = (
        best_semantic >= 0.55
    )


    # --------------------------------------------------------
    # No relevant papers
    # --------------------------------------------------------

    if not (
        has_meaningful_keyword
        or
        has_strong_tfidf
        or
        has_strong_semantic
    ):

        return {
            "query": query,
            "related_topics": [],
            "results": [],
            "message": (
                "No relevant papers found "
                "for this query."
            )
        }


    # ========================================================
    # RANK
    # ========================================================

    ranked_indices = np.argsort(
        ranking_scores
    )[::-1]


    top_indices = ranked_indices[
        :top_k
    ]


    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []


    for index in top_indices:

        paper = (
            papers.iloc[index]
        )


        result = {

            "Paper ID":
                paper["Paper ID"],

            "Title":
                paper["Title"],

            "Authors":
                paper["Authors"],

            "Published":
                paper["Published"],

            "Abstract":
                paper["Abstract"],

            "Topic":
                paper["Topic"],

            "Domain":
                paper.get(
                    "Domain",
                    ""
                ),

            "Source":
                paper.get(
                    "Source_Type",
                    ""
                ),

            "Paper URL":
                paper.get(
                    "Paper URL",
                    ""
                ),

            "Similarity":
                float(
                    semantic_scores[index]
                    * 100
                ),

            "Ranking Score":
                float(
                    ranking_scores[index]
                    * 100
                )
        }


        results.append(
            result
        )


    return {

        "query": query,

        "related_topics":
            related_topics,

        "results":
            results,

        "message":
            ""
    }