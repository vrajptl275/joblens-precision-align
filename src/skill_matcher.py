"""
skill_matcher.py — Deterministic skill matching for JobLens

Computes Matching Skills and Missing Skills by simple string intersection
between the resume text and the job's required_skills list.
This is more reliable than asking the LLM to do it.
"""

from __future__ import annotations
import re


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_skill_match(resume_text: str, required_skills: str | list) -> dict:
    """
    Check which required skills appear in the resume text.
    Handles synonym mappings and word-level content matches.
    """
    # Normalise required skills into a list
    if isinstance(required_skills, str):
        skills = [s.strip() for s in required_skills.split(",") if s.strip()]
    else:
        skills = [str(s).strip() for s in required_skills if str(s).strip()]

    resume_norm = _normalize(resume_text)
    resume_words = set(resume_norm.split())

    # Common synonym mapping (lowercased keys and list of values)
    synonyms = {
        "python": ["python development", "python programming", "python scripting"],
        "git": ["github", "gitlab", "version control"],
        "rest": ["rest api", "rest apis", "restful", "restful api", "restful apis"],
        "ml": ["machine learning", "machine learning algorithms", "machine learning models"],
        "dl": ["deep learning", "deep learning frameworks"],
        "nlp": ["natural language processing"],
        "js": ["javascript"],
        "ts": ["typescript"],
        "postgres": ["postgresql"],
        "sql": ["mysql", "sqlite", "nosql", "databases"],
        "ci/cd": ["ci cd", "ci/cd pipeline", "ci cd pipeline"],
        "aws": ["amazon web services", "sagemaker"],
        "gcp": ["google cloud", "google cloud platform"],
        "scikit-learn": ["sklearn", "scikit"]
    }

    # Build reverse synonym map
    reverse_synonyms = {}
    for key, val_list in synonyms.items():
        for val in val_list:
            reverse_synonyms[_normalize(val)] = _normalize(key)

    matching = []
    missing  = []

    # Noise words to ignore when checking sub-components
    noise_words = {
        "development", "developer", "programming", "engineer", "engineering", 
        "skills", "basics", "tools", "technologies", "methods", "methodologies", 
        "framework", "frameworks", "system", "systems", "level", "management", 
        "analysis", "analyst"
    }

    for skill in skills:
        skill_norm = _normalize(skill)
        if not skill_norm:
            continue

        # 1. Direct substring match (e.g. "sql" in "skills: python, sql")
        if skill_norm in resume_norm:
            matching.append(skill)
            continue

        # 2. Synonym mapping check
        matched_via_synonym = False
        if skill_norm in reverse_synonyms:
            base_synonym = reverse_synonyms[skill_norm]
            if base_synonym in resume_norm:
                matched_via_synonym = True
        elif skill_norm in synonyms:
            for syn in synonyms[skill_norm]:
                if _normalize(syn) in resume_norm:
                    matched_via_synonym = True
                    break

        if matched_via_synonym:
            matching.append(skill)
            continue

        # 3. Content word matching (e.g. "Python Development" -> "python")
        skill_words = [w for w in skill_norm.split() if w not in noise_words]
        if skill_words:
            if all(word in resume_words for word in skill_words):
                matching.append(skill)
                continue

        # 4. Fallback: missing
        missing.append(skill)

    return {
        "matching": matching,
        "missing":  missing,
    }


if __name__ == "__main__":
    resume = (
        "Priya Mehta, B.Tech CSE. Skills: Python, Flask, SQL, scikit-learn, basic ML. "
        "Projects: Movie Recommender, Sales Prediction."
    )
    skills = ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS SageMaker", "SQL"]
    result = compute_skill_match(resume, skills)
    print("Matching:", result["matching"])
    print("Missing: ", result["missing"])
