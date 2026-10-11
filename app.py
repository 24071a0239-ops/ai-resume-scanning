
import streamlit as st
import pandas as pd
import re
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #eef2ff, #f0fdfa);
    margin-bottom: 1.5rem;
}
.hero h1 {
    color: #172554;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #475569;
    font-size: 1.05rem;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Screening System</h1>
    <p>Screen resumes, compare skills, and discover candidates
    whose experience matches your job requirements.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1. Job requirements")

job_description = st.text_area(
    "Enter the job description",
    height=160,
    placeholder="Describe the role, responsibilities, qualifications and skills..."
)

required_skills_text = st.text_input(
    "Required skills (separate with commas)",
    value="Python, Machine Learning, SQL, Data Analysis, problem-solving"
)

st.subheader("2. Upload resumes")

uploaded_files = st.file_uploader(
    "Select one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

if st.button("🔍 Screen Resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        candidates = []
        errors = []

        with st.spinner("Analyzing resumes..."):
            for pdf in uploaded_files:
                try:
                    text = extract_pdf_text(pdf)
                    if not text.strip():
                        errors.append(
                            f"{pdf.name}: no readable text found. "
                            "Scanned PDFs may need OCR."
                        )
                        continue
                    candidates.append({
                        "Candidate": pdf.name,
                        "Text": text,
                        "Normalized": normalize(text)
                    })
                except Exception:
                    errors.append(f"{pdf.name}: could not read this PDF.")

            for error in errors:
                st.warning(error)

            if candidates:
                skills = [
                    skill.strip()
                    for skill in required_skills_text.split(",")
                    if skill.strip()
                ]

                documents = [job_description] + [
                    candidate["Text"] for candidate in candidates
                ]
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
                matrix = vectorizer.fit_transform(documents)
                scores = cosine_similarity(
                    matrix[0:1], matrix[1:]
                ).flatten()

                results = []
                for candidate, score in zip(candidates, scores):
                    matched = [
                        skill for skill in skills
                        if normalize(skill) in candidate["Normalized"]
                    ]
                    missing = [
                        skill for skill in skills
                        if normalize(skill) not in candidate["Normalized"]
                    ]

                    results.append({
                        "Candidate": candidate["Candidate"],
                        "Similarity Score (%)": round(float(score) * 100, 2),
                        "Matched Skills": ", ".join(matched) or "None",
                        "Missing Skills": ", ".join(missing) or "None",
                        "Skills Matched": len(matched),
                        "Total Required Skills": len(skills)
                    })

                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(
                    "Similarity Score (%)", ascending=False
                ).reset_index(drop=True)
                results_df.insert(0, "Rank", range(1, len(results_df) + 1))

                st.success(f"Successfully processed {len(results_df)} resume(s)!")

                st.subheader("3. Screening overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("Resumes analyzed", len(results_df))
                col2.metric(
                    "Top similarity score",
                    f"{results_df.iloc[0]['Similarity Score (%)']:.2f}%"
                )
                col3.metric(
                    "Average similarity",
                    f"{results_df['Similarity Score (%)'].mean():.2f}%"
                )

                st.subheader("4. Candidate ranking")
                chart_data = results_df.set_index("Candidate")[
                    ["Similarity Score (%)"]
                ]
                st.bar_chart(chart_data, horizontal=True)

                st.subheader("5. Detailed results")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("6. Skill match details")
                for _, row in results_df.iterrows():
                    with st.expander(
                        f"Rank {row['Rank']} — {row['Candidate']}"
                    ):
                        st.write(
                            f"**Text similarity:** "
                            f"{row['Similarity Score (%)']:.2f}%"
                        )
                        st.progress(
                            min(
                                1.0,
                                row["Skills Matched"]
                                / max(1, row["Total Required Skills"])
                            )
                        )
                        st.write(
                            f"**Skills matched:** {row['Skills Matched']} "
                            f"of {row['Total Required Skills']}"
                        )
                        st.write(
                            "**Matched skills:** "
                            + row["Matched Skills"]
                        )
                        st.write(
                            "**Missing skills:** "
                            + row["Missing Skills"]
                        )

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download screening results (CSV)",
                    data=csv_data,
                    file_name="resume_screening_results.csv",
                    mime="text/csv"
                )

                st.info(
                    "These scores measure textual similarity, not a "
                    "candidate's overall ability or suitability. Review "
                    "resumes manually before making hiring decisions."
                )
            else:
                st.error("No readable resumes were found. Please check your PDFs.")

st.caption("AI Resume Screening System • Resume analysis for decision support")
