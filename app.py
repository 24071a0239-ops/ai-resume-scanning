
import streamlit as st
import pandas as pd
import re
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="TalentSpark | AI Resume Screening",
    page_icon="💜",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --purple: #7c3aed;
    --pink: #ec4899;
    --indigo: #4f46e5;
    --ink: #211a39;
    --muted: #77738b;
}

.stApp {
    background: linear-gradient(145deg, #faf7ff 0%, #f4f0ff 48%, #fff7fc 100%);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink);
}
.hero {
    background: linear-gradient(120deg, #6d28d9 0%, #8b5cf6 48%, #ec4899 100%);
    padding: 2.2rem;
    border-radius: 25px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 14px 35px rgba(124, 58, 237, .20);
}
.hero h1 {
    color: white !important;
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    margin: .5rem 0;
}
.hero p {
    color: #f4eaff !important;
    font-size: 1.05rem;
    max-width: 720px;
}
.eyebrow {
    color: #fce7f3;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: 2px;
}
.panel {
    background: rgba(255,255,255,.88);
    border: 1px solid #ebe4fa;
    border-radius: 20px;
    padding: 1.3rem;
    margin: .7rem 0 1.1rem 0;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #ebe4fa;
    padding: 1.1rem;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(65, 35, 115, .05);
}
div[data-testid="stMetricLabel"] {
    color: #77738b;
}
div[data-testid="stMetricValue"] {
    color: #6d28d9;
    font-weight: 800;
}
.stButton > button {
    background: linear-gradient(100deg, #7c3aed, #db2777);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: .65rem 1rem;
    font-weight: 700;
}
.stButton > button:hover {
    color: white;
    border: 0;
    filter: brightness(1.06);
}
div[data-testid="stFileUploader"] {
    background: #fff;
    border: 1px dashed #b9a1f8;
    border-radius: 16px;
    padding: 1rem;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #ebe4fa;
    border-radius: 14px;
    overflow: hidden;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,.9);
    border: 1px solid #e9ddff;
    border-radius: 14px;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0eaff, #fff5fb);
}
.footer {
    text-align: center;
    color: #89829c;
    font-size: .85rem;
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ SMART RECRUITMENT WORKSPACE</div>
    <h1>Find your next great hire.</h1>
    <p>Meet TalentSpark — a simple AI-assisted workspace to compare resumes,
    explore skill matches, and organize candidates against a job description.</p>
    <span>💜 Resume insights &nbsp; ✦ Skill matching &nbsp; 🚀 Candidate ranking</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💜 TalentSpark")
    st.caption("AI Resume Screening")
    st.divider()
    st.markdown("### How it works")
    st.write("**01** · Add the job description")
    st.write("**02** · List required skills")
    st.write("**03** · Upload PDF resumes")
    st.write("**04** · Review candidate results")
    st.divider()
    st.caption(
        "Decision-support tool only. Always review resumes fairly "
        "and make hiring decisions with human judgment."
    )

st.markdown("### ✨ Let's find the right match")
st.write("Start with the role details, then upload resumes to compare.")

with st.container(border=True):
    st.markdown("#### 01 · Job description")
    job_description = st.text_area(
        "Describe the role and qualifications",
        height=150,
        placeholder="Example: We need a candidate with Python, Machine Learning, SQL, Data Analysis, and problem-solving skills.",
        label_visibility="collapsed"
    )

    st.markdown("#### 02 · Required skills")
    skills_text = st.text_input(
        "Separate skills with commas",
        value="Python, Machine Learning, SQL, Data Analysis, problem-solving",
        help="These phrases are checked in each resume.",
        label_visibility="collapsed"
    )

with st.container(border=True):
    st.markdown("#### 03 · Upload candidate resumes")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned PDFs may need OCR."
    )
    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} PDF file(s) selected")


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("PDF is password-protected.")
        except Exception as exc:
            raise ValueError("PDF is password-protected.") from exc
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def skill_is_present(skill, resume_text):
    pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
    return re.search(pattern, resume_text, flags=re.IGNORECASE) is not None


if st.button("✨ Analyze & Rank Resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a job description first.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        skills = list(dict.fromkeys(
            skill.strip() for skill in skills_text.split(",") if skill.strip()
        ))
        candidates = []
        errors = []

        with st.spinner("✨ Reading resumes and comparing details..."):
            for pdf in uploaded_files:
                try:
                    text = extract_pdf_text(pdf)
                    if not text.strip():
                        errors.append(
                            f"{pdf.name}: no extractable text. "
                            "Scanned PDFs may need OCR."
                        )
                        continue
                    candidates.append({
                        "Candidate": pdf.name,
                        "Text": text
                    })
                except Exception as exc:
                    errors.append(f"{pdf.name}: could not read PDF ({exc}).")

            for message in errors:
                st.warning(message)

            if not candidates:
                st.error("No readable resumes were found. Please check the PDFs.")
            else:
                documents = [job_description] + [
                    candidate["Text"] for candidate in candidates
                ]
                try:
                    vectorizer = TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2)
                    )
                    matrix = vectorizer.fit_transform(documents)
                    scores = cosine_similarity(
                        matrix[0:1], matrix[1:]
                    ).flatten()
                except ValueError:
                    scores = [0.0] * len(candidates)
                    st.warning(
                        "Not enough usable text to calculate similarity; "
                        "scores are shown as 0%."
                    )

                results = []
                for candidate, score in zip(candidates, scores):
                    matched = [
                        skill for skill in skills
                        if skill_is_present(skill, candidate["Text"])
                    ]
                    missing = [
                        skill for skill in skills
                        if not skill_is_present(skill, candidate["Text"])
                    ]
                    skill_pct = (
                        round(len(matched) / len(skills) * 100, 2)
                        if skills else 0.0
                    )
                    results.append({
                        "Candidate": candidate["Candidate"],
                        "Similarity Score (%)": round(float(score) * 100, 2),
                        "Skills Matched": f"{len(matched)}/{len(skills)}",
                        "Skill Match (%)": skill_pct,
                        "Matched Skills": ", ".join(matched) or "None",
                        "Missing Skills": ", ".join(missing) or "None"
                    })

                df = pd.DataFrame(results).sort_values(
                    "Similarity Score (%)", ascending=False
                ).reset_index(drop=True)
                df.insert(0, "Rank", range(1, len(df) + 1))

                st.markdown("---")
                st.markdown("### 📊 Your screening snapshot")
                c1, c2, c3 = st.columns(3)
                c1.metric("📄 Resumes analyzed", len(df))
                c2.metric("🏆 Top similarity", f"{df.iloc[0]['Similarity Score (%)']:.2f}%")
                c3.metric("✨ Average similarity", f"{df['Similarity Score (%)'].mean():.2f}%")

                st.markdown("### 🏅 Candidate leaderboard")
                st.caption("Ranked by textual similarity to the job description.")
                st.bar_chart(
                    df.set_index("Candidate")[["Similarity Score (%)"]],
                    horizontal=True
                )

                st.markdown("### 🧾 Candidate report")
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("### 💎 Skill match explorer")
                for _, row in df.iterrows():
                    with st.expander(
                        f"#{row['Rank']}  ·  {row['Candidate']}  ·  "
                        f"{row['Skill Match (%)']:.0f}% skill match"
                    ):
                        left, right = st.columns(2)
                        left.metric("Text similarity", f"{row['Similarity Score (%)']:.2f}%")
                        right.metric("Required skills matched", row["Skills Matched"])
                        st.progress(int(row["Skill Match (%)"]) / 100)
                        st.markdown("**🟣 Matched skills**")
                        st.write(row["Matched Skills"])
                        st.markdown("**🩷 Skills to review**")
                        st.write(row["Missing Skills"])

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download candidate report (CSV)",
                    data=csv_data,
                    file_name="talentspark_screening_results.csv",
                    mime="text/csv"
                )

                st.info(
                    "Similarity scores measure textual overlap, not overall "
                    "candidate ability. Skill checks use exact phrases and may "
                    "miss equivalent wording. Review all candidates fairly; "
                    "do not use this tool as the sole basis for hiring."
                )

st.markdown("""
<div class="footer">
    💜 TalentSpark · AI-assisted resume screening · Built for human-led decisions
</div>
""", unsafe_allow_html=True)
