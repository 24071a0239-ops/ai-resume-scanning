
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
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #eef2ff, #f0fdfa);
    margin-bottom: 1.5rem;
}
.hero h1 {color: #172554; margin-bottom: .4rem;}
.hero p {color: #475569; font-size: 1.05rem;}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📄 AI Resume Screening System</h1>
<p>Compare resumes with job requirements, review skill matches,
and organize candidates for further evaluation.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("1. Job requirements")

job_description = st.text_area(
    "Enter the job description",
    height=160,
    placeholder="Describe the role, qualifications, and required skills..."
)

skills_text = st.text_input(
    "Required skills (comma-separated)",
    value="Python, Machine Learning, SQL, Data Analysis, problem-solving"
)

st.subheader("2. Upload resumes")

uploaded_files = st.file_uploader(
    "Select PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Text-based PDFs work best. Scanned PDFs may need OCR."
)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("This PDF is password-protected.")
        except Exception as exc:
            raise ValueError("This PDF is password-protected.") from exc

    return "\n".join(page.extract_text() or "" for page in reader.pages)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def skill_is_present(skill, resume_text):
    # Match whole words/phrases instead of partial words.
    # For example, "SQL" won't match inside "NoSQL".
    pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
    return re.search(pattern, resume_text, flags=re.IGNORECASE) is not None


if st.button("🔍 Screen Resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        skills = list(dict.fromkeys(
            skill.strip()
            for skill in skills_text.split(",")
            if skill.strip()
        ))

        candidates = []
        errors = []

        with st.spinner("Reading and comparing resumes..."):
            for pdf in uploaded_files:
                try:
                    resume_text = extract_pdf_text(pdf)

                    if not resume_text.strip():
                        errors.append(
                            f"{pdf.name}: no extractable text found. "
                            "If this is a scanned PDF, OCR may be needed."
                        )
                        continue

                    candidates.append({
                        "Candidate": pdf.name,
                        "Text": resume_text,
                        "Normalized": normalize(resume_text)
                    })

                except Exception as exc:
                    errors.append(f"{pdf.name}: could not read this PDF ({exc}).")

            for message in errors:
                st.warning(message)

            if not candidates:
                st.error("No readable resumes were found. Check the PDF files.")
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
                        "Not enough usable text to calculate similarity. "
                        "Similarity scores have been set to 0%."
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

                    skill_percent = (
                        round(len(matched) / len(skills) * 100, 2)
                        if skills else 0.0
                    )

                    results.append({
                        "Candidate": candidate["Candidate"],
                        "Similarity Score (%)": round(float(score) * 100, 2),
                        "Skills Matched": f"{len(matched)}/{len(skills)}",
                        "Skill Match (%)": skill_percent,
                        "Matched Skills": ", ".join(matched) or "None",
                        "Missing Skills": ", ".join(missing) or "None"
                    })

                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(
                    "Similarity Score (%)", ascending=False
                ).reset_index(drop=True)
                results_df.insert(0, "Rank", range(1, len(results_df) + 1))

                st.success(
                    f"Screening complete: {len(results_df)} readable "
                    f"resume(s) processed."
                )

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
                st.bar_chart(
                    results_df.set_index("Candidate")[["Similarity Score (%)"]],
                    horizontal=True
                )

                st.subheader("5. Detailed results")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                st.subheader("6. Skill match details")
                for _, row in results_df.iterrows():
                    with st.expander(
                        f"Rank {row['Rank']} — {row['Candidate']}"
                    ):
                        st.write(
                            f"**Text similarity:** "
                            f"{row['Similarity Score (%)']:.2f}%"
                        )
                        st.write(
                            f"**Required skills matched:** "
                            f"{row['Skills Matched']}"
                        )
                        st.write(
                            f"**Skill match percentage:** "
                            f"{row['Skill Match (%)']:.2f}%"
                        )
                        st.progress(
                            int(row["Skill Match (%)"]) / 100
                        )
                        st.write(f"**Matched:** {row['Matched Skills']}")
                        st.write(f"**Missing:** {row['Missing Skills']}")

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download full results (CSV)",
                    data=csv_data,
                    file_name="resume_screening_results.csv",
                    mime="text/csv"
                )

                st.info(
                    "Important: Similarity and skill-match percentages are "
                    "screening aids, not measures of overall candidate ability. "
                    "Skill checks use exact phrases and may miss equivalent "
                    "skills or alternative wording. Review resumes manually."
                )

st.caption(
    "AI Resume Screening System • Decision-support tool, not an automated "
    "hiring decision"
)
