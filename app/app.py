import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
from html import escape
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSS_PATH = BASE_DIR / "static" / "style.css"
HTML_PATH = BASE_DIR / "templates" / "results.html"

sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.ml_model import predict_topic_with_confidence
from src.ese_analysis import analyze_abstract


# Search is imported only when actually used.
# This keeps the main app startup cleaner.
def run_search(query):
    from src.search_engine import search_papers
    return search_papers(query, top_k=5)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResearchInsight",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD FRONTEND FILES
# ============================================================

with open(CSS_PATH, "r", encoding="utf-8") as f:
    css = f.read()

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html_template = f.read()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_page" not in st.session_state:
    st.session_state.analysis_page = None

if "analysis_sentence_count" not in st.session_state:
    st.session_state.analysis_sentence_count = 0

if "search_result" not in st.session_state:
    st.session_state.search_result = None


# ============================================================
# STREAMLIT PAGE CSS
# ============================================================

st.markdown(
    f"""
<style>

{css}

/* ============================================================
   GENERAL
   ============================================================ */

.block-container {{
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}


/* ============================================================
   HEADER
   ============================================================ */

.input-brand {{
    font-size: 25px;
    font-weight: 750;
    color: #172033;
    letter-spacing: -0.7px;
}}

.input-tagline {{
    color: #8a94a6;
    font-size: 13px;
    margin-top: 4px;
}}

.input-modules {{
    text-align: right;
    padding-top: 7px;
    color: #8a94a6;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}}


/* ============================================================
   SEARCH
   ============================================================ */

.search-eyebrow {{
    color: #8a94a6;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 7px;
}}

.search-title {{
    color: #172033;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}}

.search-description {{
    color: #536176;
    font-size: 14px;
    line-height: 1.6;
    max-width: 760px;
    margin-bottom: 20px;
}}

.search-divider {{
    border-top: 1px solid #e1e6ed;
    margin-top: 55px;
    margin-bottom: 45px;
}}


/* ============================================================
   SEARCH INPUT
   ============================================================ */

.stTextInput input {{
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #dfe4eb !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}}

.stTextInput input::placeholder {{
    color: #a4acb9 !important;
}}


/* ============================================================
   ABSTRACT INPUT
   ============================================================ */

.stTextArea textarea {{
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #dfe4eb !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}}

.stTextArea textarea::placeholder {{
    color: #a4acb9 !important;
}}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton button {{
    border-radius: 9px !important;
    font-weight: 650 !important;
    min-height: 43px !important;
}}


/* ============================================================
   SEARCH TOPICS
   ============================================================ */

.topic-label {{
    color: #8a94a6;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 30px;
    margin-bottom: 10px;
}}

.topic-pill {{
    display: inline-block;
    padding: 7px 12px;
    margin-right: 6px;
    margin-bottom: 6px;
    border-radius: 20px;
    background: #f1f4f8;
    border: 1px solid #e1e6ed;
    color: #344054;
    font-size: 12px;
    font-weight: 600;
}}


/* ============================================================
   SEARCH RESULTS
   ============================================================ */

.search-result-card {{
    background: #ffffff;
    border: 1px solid #e1e6ed;
    border-radius: 14px;
    padding: 22px 24px;
    margin: 12px 0;
    box-shadow: 0 3px 14px rgba(30, 45, 65, 0.045);
}}

.search-result-title {{
    color: #172033;
    font-size: 17px;
    font-weight: 700;
    line-height: 1.45;
    margin-bottom: 8px;
}}

.search-result-meta {{
    color: #8a94a6;
    font-size: 11px;
    margin-bottom: 14px;
}}

.search-result-topic {{
    display: inline-block;
    background: #f1f4f8;
    color: #344054;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .7px;
    text-transform: uppercase;
}}

.search-result-score {{
    color: #68758a;
    font-size: 11px;
    font-weight: 650;
    margin-left: 7px;
}}

.search-result-abstract {{
    color: #536176;
    font-size: 13px;
    line-height: 1.65;
    margin-top: 15px;
    margin-bottom: 17px;
}}

.paper-link {{
    display: inline-block;
    color: #172033 !important;
    text-decoration: none !important;
    border: 1px solid #dfe4eb;
    border-radius: 8px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 700;
}}

.paper-link:hover {{
    background: #f6f8fa;
}}


/* ============================================================
   NO RESULTS
   ============================================================ */

.no-results-box {{
    background: #fafbfc;
    border: 1px solid #e1e6ed;
    border-radius: 12px;
    padding: 25px;
    margin-top: 15px;
}}

.no-results-title {{
    color: #344054;
    font-size: 15px;
    font-weight: 700;
}}

.no-results-text {{
    color: #8a94a6;
    font-size: 13px;
    margin-top: 5px;
}}


/* ============================================================
   ANALYSIS HERO
   ============================================================ */

.input-hero-title {{
    font-size: 40px;
    font-weight: 700;
    color: #172033;
    letter-spacing: -1.2px;
    margin-top: 20px;
    margin-bottom: 10px;
}}

.input-hero-text {{
    color: #536176;
    font-size: 15px;
    line-height: 1.6;
    max-width: 720px;
    margin-bottom: 24px;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

left, right = st.columns([3, 1])

with left:

    st.markdown(
        '<div class="input-brand">ResearchInsight</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="input-tagline">'
        'Intelligent research paper analysis'
        '</div>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        '<div class="input-modules">'
        'ML&nbsp;&nbsp;·&nbsp;&nbsp;ESE&nbsp;&nbsp;·&nbsp;&nbsp;SEARCH'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# ============================================================
# SEARCH SECTION — TOP OF PAGE
# ============================================================
# ============================================================

st.markdown(
    '<div class="search-eyebrow">Research Library</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="search-title">'
    'Find relevant research papers.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="search-description">'
    'Search across the research corpus using topics, concepts '
    'or research questions. Results are ranked using semantic '
    'and keyword-based similarity.'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# SEARCH INPUT
# ------------------------------------------------------------

search_query = st.text_input(
    "Search research papers",
    placeholder="e.g. deep learning for medical diagnosis",
    label_visibility="collapsed",
    key="search_input"
)


# ------------------------------------------------------------
# SEARCH BUTTON
# ------------------------------------------------------------

search_col, _ = st.columns([1, 5])

with search_col:

    search_button = st.button(
        "Search Papers →",
        type="primary",
        use_container_width=True,
        key="search_button"
    )


# ------------------------------------------------------------
# RUN SEARCH
# ------------------------------------------------------------

if search_button:

    if not search_query.strip():

        st.warning(
            "Please enter a search query first."
        )

    else:

        with st.spinner(
            "Searching research papers..."
        ):

            try:

                st.session_state.search_result = (
                    run_search(
                        search_query.strip()
                    )
                )

            except Exception as error:

                st.session_state.search_result = None

                st.error(
                    "Unable to perform the search."
                )

                st.caption(str(error))


# ------------------------------------------------------------
# DISPLAY SEARCH RESULTS
# ------------------------------------------------------------

if st.session_state.search_result is not None:

    search_result = (
        st.session_state.search_result
    )


    # ========================================================
    # RELATED RESEARCH AREAS
    # ========================================================

    related_topics = search_result.get(
        "related_topics",
        []
    )


    if related_topics:

        st.markdown(
            '<div class="topic-label">'
            'Related Research Areas'
            '</div>',
            unsafe_allow_html=True
        )


        topic_html = ""

        for topic_name, score in related_topics[:5]:

            topic_html += (
                '<span class="topic-pill">'
                +
                escape(
                    str(topic_name).title()
                )
                +
                '</span>'
            )


        st.markdown(
            topic_html,
            unsafe_allow_html=True
        )


    # ========================================================
    # PAPERS
    # ========================================================

    results = search_result.get(
        "results",
        []
    )


    if results:

        st.markdown(
            '<div class="topic-label">'
            'Relevant Papers'
            '</div>',
            unsafe_allow_html=True
        )


        for i, paper in enumerate(
            results,
            start=1
        ):

            # ------------------------------------------------
            # BASIC DATA
            # ------------------------------------------------

            title = escape(
                str(
                    paper.get(
                        "Title",
                        "Untitled Paper"
                    )
                )
            )


            topic_name = escape(
                str(
                    paper.get(
                        "Topic",
                        "Unknown"
                    )
                ).title()
            )


            authors = escape(
                str(
                    paper.get(
                        "Authors",
                        ""
                    )
                )
            )


            published = escape(
                str(
                    paper.get(
                        "Published",
                        ""
                    )
                )
            )


            abstract_text = escape(
                str(
                    paper.get(
                        "Abstract",
                        ""
                    )
                )
            )


            if abstract_text.lower() == "nan":

                abstract_text = ""


            if len(abstract_text) > 420:

                abstract_text = (
                    abstract_text[:420]
                    + "..."
                )


            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            try:

                similarity = float(
                    paper.get(
                        "Similarity",
                        0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                similarity = 0.0


            # ------------------------------------------------
            # PAPER URL
            # ------------------------------------------------

            url = str(
                paper.get(
                    "Paper URL",
                    ""
                )
            ).strip()


            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            metadata_parts = []

            if (
                authors
                and authors.lower() != "nan"
            ):

                metadata_parts.append(
                    authors
                )


            if (
                published
                and published.lower() != "nan"
            ):

                metadata_parts.append(
                    published
                )


            metadata = (
                " · ".join(
                    metadata_parts
                )
            )


            # ------------------------------------------------
            # LINK
            # ------------------------------------------------

            paper_link = ""

            if (
                url.startswith("http://")
                or
                url.startswith("https://")
            ):

                paper_link = (
                    '<a class="paper-link" href="'
                    +
                    escape(
                        url,
                        quote=True
                    )
                    +
                    '" target="_blank">'
                    'View Paper ↗'
                    '</a>'
                )


            # ------------------------------------------------
            # CARD
            # ------------------------------------------------

            card_html = (
                '<div class="search-result-card">'

                '<div class="search-result-title">'
                + str(i)
                + '. '
                + title
                + '</div>'

                '<div class="search-result-meta">'
                + metadata
                + '</div>'

                '<span class="search-result-topic">'
                + topic_name
                + '</span>'

                '<span class="search-result-score">'
                + f"{similarity:.1f}% semantic relevance"
                + '</span>'

                '<div class="search-result-abstract">'
                + abstract_text
                + '</div>'

                + paper_link

                + '</div>'
            )


            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


    else:

        st.markdown(
        '<div class="no-results-box">'
        '<div class="no-results-title">'
        'No relevant papers found'
        '</div>'
        '<div class="no-results-text">'
        'Try a broader research concept or a different search term.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DIVIDER BEFORE ABSTRACT ANALYSIS
# ============================================================

st.markdown(
    '<div class="search-divider"></div>',
    unsafe_allow_html=True
)


# ============================================================
# ============================================================
# ABSTRACT ANALYSIS SECTION
# ============================================================
# ============================================================

st.markdown(
    '<div class="input-hero-title">'
    'Analyze a research abstract.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-hero-text">'
    'Discover the research topic, structure, findings and key '
    'aspects of an abstract using machine learning and natural '
    'language analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABSTRACT INPUT
# ============================================================

abstract = st.text_area(
    "Research Abstract",
    placeholder="Paste your research abstract here...",
    height=250,
    label_visibility="collapsed",
    key="abstract_input"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_col, _ = st.columns([1, 5])

with analyze_col:

    analyze_button = st.button(
        "Analyze Abstract →",
        type="primary",
        use_container_width=True,
        key="analyze_button"
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_button:

    if not abstract.strip():

        st.warning(
            "Please paste a research abstract first."
        )

    else:

        with st.spinner(
            "Analyzing research abstract..."
        ):

            # ------------------------------------------------
            # MACHINE LEARNING
            # ------------------------------------------------

            topic, confidence = (
                predict_topic_with_confidence(
                    abstract
                )
            )


            # ------------------------------------------------
            # ESE
            # ------------------------------------------------

            ese_result = analyze_abstract(
                abstract
            )


        # ====================================================
        # DOCUMENT LEVEL
        # ====================================================

        document = ese_result["Document Level"]


        objective = (
            document["Objective"][0]
            if document["Objective"]
            else "Not explicitly identified."
        )


        method = (
            document["Method"][0]
            if document["Method"]
            else "Not explicitly identified."
        )


        findings = document["Findings"][:2]

        conclusions = document["Conclusion"][:2]


        # ====================================================
        # FINDINGS
        # ====================================================

        if findings:

            findings_html = ""

            for finding in findings:

                findings_html += (
                    '<div class="finding">'
                    +
                    escape(str(finding))
                    +
                    '</div>'
                )

        else:

            findings_html = (
                '<span>Not explicitly identified.</span>'
            )


        # ====================================================
        # CONCLUSIONS
        # ====================================================

        if conclusions:

            conclusion_html = ""

            for conclusion in conclusions:

                conclusion_html += (
                    '<div class="finding">'
                    +
                    escape(str(conclusion))
                    +
                    '</div>'
                )

        else:

            conclusion_html = (
                '<span>Not explicitly identified.</span>'
            )


        # ====================================================
        # ASPECTS
        # ====================================================

        aspects = ese_result["Aspect Level"]

        unique_aspects = []

        seen = set()


        for item in aspects:

            name = item["Aspect"]

            if name not in seen:

                seen.add(name)

                unique_aspects.append(item)


        unique_aspects = unique_aspects[:6]


        aspects_html = ""


        for item in unique_aspects:

            aspects_html += f"""
<div class="aspect-card">

    <div class="aspect-name">
        {escape(str(item["Aspect"]))}
    </div>

    <div class="aspect-evidence">
        {escape(str(item["Evidence"]))}
    </div>

    <div class="aspect-stance">
        {escape(str(item["Research Stance"]))}
    </div>

</div>
"""


        if not aspects_html:

            aspects_html = """
<div class="aspect-card">

    <div class="aspect-name">
        No major aspects identified
    </div>

    <div class="aspect-evidence">
        No major research aspects were identified.
    </div>

</div>
"""


        # ====================================================
        # SENTENCE LEVEL
        # ====================================================

        sentence_results = ese_result[
            "Sentence Level"
        ]


        sentences_html = ""


        for item in sentence_results:

            sentences_html += f"""
<div class="sentence">

    <div class="sentence-number">
        {escape(str(item["Sentence No"]))}
    </div>

    <div>
        <span class="sentence-role">
            {escape(str(item["Role"]))}
        </span>
    </div>

    <div class="sentence-text">
        {escape(str(item["Sentence"]))}
    </div>

</div>
"""


        # ====================================================
        # ABSTRACT STRUCTURE
        # ====================================================

        roles = [
            item["Role"]
            for item in sentence_results
        ]


        simplified_roles = []


        for role in roles:

            if (
                not simplified_roles
                or simplified_roles[-1] != role
            ):

                simplified_roles.append(role)


        structure_html = ""


        for i, role in enumerate(
            simplified_roles
        ):

            structure_html += (
                '<span class="structure-item">'
                +
                escape(str(role))
                +
                '</span>'
            )


            if i < len(simplified_roles) - 1:

                structure_html += (
                    '<span class="arrow">→</span>'
                )


        # ====================================================
        # BUILD RESULT PAGE
        # ====================================================

        page = html_template


        # ----------------------------------------------------
        # CRITICAL FIX
        #
        # Always inject CSS directly into the iframe.
        # This prevents the result page from falling back
        # to ugly default browser formatting.
        # ----------------------------------------------------

        result_css = f"""
<style>

{css}

/* ==========================================================
   RESULT PAGE FALLBACK STYLING
   ========================================================== */

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
}}

body {{
    background: #F7F8FA !important;
    color: #172033 !important;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}}

h1,
h2,
h3,
h4,
p,
div,
span {{
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

.result-container {{
    max-width: 1100px;
    margin: auto;
}}

.result-box {{
    background: #ffffff;
    border: 1px solid #E1E5EB;
    border-radius: 12px;
    padding: 22px;
}}

.aspect-card {{
    background: #ffffff;
    border: 1px solid #E1E5EB;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
}}

.sentence {{
    background: #ffffff;
    border: 1px solid #E1E5EB;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
}}

.sentence-number {{
    color: #8A94A6;
    font-size: 11px;
    font-weight: 700;
}}

.sentence-role {{
    display: inline-block;
    background: #F1F4F8;
    color: #344054;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 700;
    margin: 5px 0 8px 0;
}}

.sentence-text {{
    color: #536176;
    font-size: 13px;
    line-height: 1.65;
}}

.aspect-name {{
    color: #172033;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 9px;
}}

.aspect-evidence {{
    color: #536176;
    font-size: 13px;
    line-height: 1.55;
}}

.aspect-stance {{
    color: #8A94A6;
    font-size: 11px;
    margin-top: 10px;
}}

.finding {{
    background: #F8FAFC;
    border-left: 3px solid #D9DFE8;
    padding: 10px 13px;
    margin: 7px 0;
    color: #536176;
    font-size: 13px;
    line-height: 1.55;
}}

.structure-item {{
    display: inline-block;
    background: #F1F4F8;
    color: #344054;
    border: 1px solid #E1E6ED;
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 11px;
    font-weight: 700;
}}

.arrow {{
    color: #9AA2B1;
    margin: 0 8px;
}}

</style>
"""


        # ----------------------------------------------------
        # Inject CSS BEFORE HTML
        # ----------------------------------------------------

        if "{{CSS}}" in page:

            page = page.replace(
                "{{CSS}}",
                css
            )

        else:

            page = (
                result_css
                +
                page
            )


        # ----------------------------------------------------
        # DATA PLACEHOLDERS
        # ----------------------------------------------------

        page = page.replace(
            "{{TOPIC}}",
            escape(str(topic).title())
        )


        page = page.replace(
            "{{OBJECTIVE}}",
            escape(str(objective))
        )


        page = page.replace(
            "{{METHOD}}",
            escape(str(method))
        )


        page = page.replace(
            "{{FINDINGS}}",
            findings_html
        )


        page = page.replace(
            "{{CONCLUSION}}",
            conclusion_html
        )


        page = page.replace(
            "{{ASPECTS}}",
            aspects_html
        )


        page = page.replace(
            "{{ASPECT_COUNT}}",
            str(len(unique_aspects))
        )


        page = page.replace(
            "{{SENTENCES}}",
            sentences_html
        )


        page = page.replace(
            "{{SENTENCE_COUNT}}",
            str(len(sentence_results))
        )


        page = page.replace(
            "{{STRUCTURE}}",
            structure_html
        )


        # ----------------------------------------------------
        # STORE RESULT IN SESSION STATE
        # ----------------------------------------------------

        st.session_state.analysis_page = page

        st.session_state.analysis_sentence_count = (
            len(sentence_results)
        )


# ============================================================
# DISPLAY ANALYSIS RESULT
# ============================================================

if st.session_state.analysis_page is not None:

    height = max(
        1100,
        700
        +
        st.session_state.analysis_sentence_count * 85
    )


    components.html(
        st.session_state.analysis_page,
        height=height,
        scrolling=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#a4acb9;
        font-size:11px;
        margin:60px 0 25px 0;
        letter-spacing:.3px;
    ">
        ResearchInsight · Intelligent Research Analysis
    </div>
    """,
    unsafe_allow_html=True
)