
import streamlit as st
import pandas as pd
import re
from io import BytesIO
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.write(
    "Upload PDF resumes, compare them with a job description, "
    "and rank candidates by text similarity and required skills."
)

job_description = st.text_area(
    "Enter the Job Description",
    height=180,
    placeholder=(
        "Example: We need a candidate with Python, "
        "Machine Learning, SQL, Data Analysis, and problem-solving skills."
    )
)

required_skills_text = st.text_input(
    "Required skills (comma-separated)",
    value="Python, Machine Learning, SQL, Data Analysis, problem-solving"
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


if st.button("🔍 Screen Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        try:
            with st.spinner("Reading and comparing resumes..."):
                skills = [
                    skill.strip()
                    for skill in required_skills_text.split(",")
                    if skill.strip()
                ]

                candidates = []

                for pdf in uploaded_files:
                    resume_text = extract_pdf_text(pdf)

                    if not resume_text.strip():
                        st.warning(
                            f"{pdf.name} contains no extractable text. "
                            "It may be a scanned PDF."
                        )
                        continue

                    candidates.append({
                        "Candidate": pdf.name,
                        "Resume Text": resume_text,
                        "Normalized Text": normalize(resume_text)
                    })

                if not candidates:
                    st.error(
                        "No readable resume text was found. "
                        "Try text-based PDFs rather than scanned images."
                    )
                else:
                    documents = [job_description] + [
                        candidate["Resume Text"]
                        for candidate in candidates
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
                        resume = candidate["Normalized Text"]

                        matched = [
                            skill for skill in skills
                            if normalize(skill) in resume
                        ]
                        missing = [
                            skill for skill in skills
                            if normalize(skill) not in resume
                        ]

                        results.append({
                            "Candidate": candidate["Candidate"],
                            "Similarity Score (%)": round(
                                float(score) * 100, 2
                            ),
                            "Matched Skills": ", ".join(matched) or "None",
                            "Missing Skills": ", ".join(missing) or "None",
                            "Skills Matched": f"{len(matched)}/{len(skills)}"
                        })

                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values(
                        by="Similarity Score (%)",
                        ascending=False
                    ).reset_index(drop=True)

                    results_df.insert(
                        0, "Rank", range(1, len(results_df) + 1)
                    )

                    st.success(
                        f"Screening complete! "
                        f"{len(results_df)} resume(s) processed."
                    )

                    st.subheader("🏆 Ranked Candidates")
                    st.dataframe(results_df, use_container_width=True)

                    best = results_df.iloc[0]
                    st.metric(
                        "Top similarity score",
                        f"{best['Similarity Score (%)']:.2f}%",
                        label_visibility="visible"
                    )
                    st.write(f"**Top-ranked file:** {best['Candidate']}")

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results as CSV",
                        data=csv_data,
                        file_name="resume_screening_results.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "Note: Similarity scores measure textual overlap, "
                        "not a candidate's true suitability. Review resumes "
                        "manually before making hiring decisions."
                    )

        except Exception as error:
            st.error(f"An error occurred while screening resumes: {error}")
