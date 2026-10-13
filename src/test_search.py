from search_engine import search_papers


# ============================================================
# TEST SEARCH
# ============================================================

query = (
    "deep learning for medical diagnosis"
)


result = search_papers(
    query,
    top_k=5
)


print(
    "\nQUERY:",
    result["query"]
)


print(
    "\nRELATED TOPICS:"
)

for topic, score in (
    result["related_topics"]
):

    print(
        f"{topic}: {score:.4f}"
    )


if not result["results"]:

    print(
        "\n⚠️",
        result["message"]
    )

else:

    print(
        "\nTOP PAPERS:\n"
    )


    for i, paper in enumerate(
        result["results"],
        start=1
    ):

        print(
            f"{i}. {paper['Title']}"
        )

        print(
            f"   Topic: "
            f"{paper['Topic']}"
        )

        print(
            f"   Similarity: "
            f"{paper['Similarity']:.2f}%"
        )

        print(
            f"   URL: "
            f"{paper['Paper URL']}"
        )

        print()