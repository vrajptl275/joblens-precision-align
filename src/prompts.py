"""
prompts.py — Prompt-building logic for JobLens

Skills are computed deterministically by skill_matcher.py and injected
into the prompt. The LLM only handles: Match Score and Suggestion —
matching the exact format it was fine-tuned on.
"""

from __future__ import annotations
from src.skill_matcher import compute_skill_match


# ── System instruction ─────────────────────────────────────────────────────────
# Not currently sent to Ollama (the /api/generate prompt endpoint doesn't use it).
# Kept here for reference / future use if llm_client.py is switched to /api/chat.
SYSTEM_PROMPT = (
    "You are a career advisor AI. "
    "Evaluate how well a candidate's resume matches a job posting. "
    "Always respond in the exact format requested — no extra commentary."
)


# ── Per-job prompt ─────────────────────────────────────────────────────────────

def build_match_prompt(resume_text: str, job: dict) -> str:
    """
    Build the prompt for a single resume <-> job comparison.

    Matching/missing skills are pre-computed by skill_matcher.py and injected
    into the prompt so the LLM only needs to assess:
      - Match Score
      - Suggestion

    NOTE: This prompt intentionally mirrors the model's fine-tuning format
    (`Resume: ... \\nJob: ...`) as closely as possible. The model was trained
    on exactly four output fields (Match Score, Matching Skills, Missing
    Skills, Suggestion) — asking for anything beyond that (e.g. Experience
    Match, Location Note) produces unreliable output since the model never
    learned those fields during training.
    """
    raw_skills = job.get("required_skills", "")
    if isinstance(raw_skills, list):
        skills_list = [s.strip() for s in raw_skills if s.strip()]
        skills_str  = ", ".join(skills_list)
    else:
        skills_str  = raw_skills
        skills_list = [s.strip() for s in raw_skills.split(",") if s.strip()]

    title    = job.get("title", "Unknown Role")
    company  = job.get("company", "Unknown Company")
    location = job.get("location", "Not specified")
    exp      = job.get("experience_level", "Not specified")

    job_description = job.get("document") or (
        f"{title} at {company}. Requires {skills_str}. "
        f"Location: {location}. Experience: {exp}."
    )

    # ── Deterministic skill matching ───────────────────────────────────────────
    skill_result = compute_skill_match(resume_text, skills_list)
    matched_str  = ", ".join(skill_result["matching"]) if skill_result["matching"] else "None"
    missing_str  = ", ".join(skill_result["missing"])  if skill_result["missing"]  else "None"

    prompt = f"""Resume: {resume_text.strip()}
Job: {job_description.strip()}

Skill analysis (already computed — use exactly as written):
Matching Skills: {matched_str}
Missing Skills: {missing_str}

Respond in this EXACT format:
Match Score: <number>/100
Matching Skills: {matched_str}
Missing Skills: {missing_str}
Suggestion: <one actionable sentence>"""

    return prompt


# ── Batch helper ───────────────────────────────────────────────────────────────

def build_all_prompts(resume_text: str, jobs: list[dict]) -> list[str]:
    """Build one prompt per retrieved job."""
    return [build_match_prompt(resume_text, job) for job in jobs]


if __name__ == "__main__":
    sample_resume = (
        "Priya Mehta, B.Tech CSE, Final Year.\n"
        "Skills: Python, Flask, SQL, scikit-learn, basic ML.\n"
        "Projects: Movie Recommender, Sales Prediction.\n"
        "Location: Mumbai. Experience: Fresher."
    )
    sample_job = {
        "title":            "Senior Data Scientist",
        "company":          "AnalyticsHub",
        "location":         "Mumbai",
        "experience_level": "5-8 years",
        "required_skills":  ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS SageMaker", "SQL"],
    }

    print(build_match_prompt(sample_resume, sample_job))