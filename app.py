from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Common technical skills for comparison
SKILLS_DATABASE = [
    "python", "java", "c", "c++", "sql", "mysql", "mongodb",
    "html", "css", "javascript", "bootstrap", "react", "angular",
    "nodejs", "django", "flask", "spring", "git", "github",
    "docker", "kubernetes", "aws", "azure", "machine learning",
    "deep learning", "artificial intelligence", "data science",
    "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
    "power bi", "excel", "tableau", "rest api", "api", "linux",
    "oop", "problem solving", "communication", "teamwork"
]


def preprocess(text):
    """Convert text to lowercase and remove extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s#+.]', ' ', text)
    return text


def extract_skills(text):
    """Extract known skills from text."""
    text = preprocess(text)
    found = []

    for skill in SKILLS_DATABASE:
        if skill.lower() in text:
            found.append(skill.title())

    return sorted(list(set(found)))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    resume = request.form.get("resume", "")
    job = request.form.get("job", "")

    if resume.strip() == "" or job.strip() == "":
        return render_template(
            "result.html",
            has_analysis=False,
            match=0,
            ats=0,
            resume_skills=[],
            job_skills=[],
            missing_skills=[],
            suggestions=["Please provide both resume text and a job description so I can generate your analysis."]
        )

    # ---------- Resume Match ----------
    documents = [resume, job]

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(matrix[0], matrix[1])[0][0]

    match_percentage = round(similarity * 100, 2)

    # ---------- ATS Score ----------
    ats_score = min(100, round(match_percentage + 10))

    # ---------- Skill Extraction ----------
    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job)

    missing_skills = sorted(
        list(set(job_skills) - set(resume_skills))
    )

    # ---------- Suggestions ----------
    suggestions = []

    if match_percentage >= 80:
        suggestions.append("Excellent fit. Your profile is strongly aligned with the role.")
    elif match_percentage >= 60:
        suggestions.append("Strong foundation. A few targeted updates can make your application even sharper.")
    else:
        suggestions.append("The alignment is still developing. Tailoring your resume language to the role will help.")

    if missing_skills:
        suggestions.append("Highlight any relevant experience you have with these skills in your summary or achievements.")

    suggestions.append("Add measurable wins such as improved accuracy, reduced time, or revenue impact.")
    suggestions.append("Keep the document concise and easy to scan for recruiters and ATS systems.")
    suggestions.append("Use keywords from the job description naturally throughout your resume.")

    return render_template(
        "result.html",
        has_analysis=True,
        match=match_percentage,
        ats=ats_score,
        resume_skills=resume_skills,
        job_skills=job_skills,
        missing_skills=missing_skills,
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)