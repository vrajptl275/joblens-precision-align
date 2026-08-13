"""
retrieve.py — Semantic retrieval from ChromaDB for JobLens

Given plain resume text, returns the top-K most semantically similar
job postings from the pre-built ChromaDB index — filtered so that jobs
requiring dramatically more experience than the candidate has don't crowd
out realistic matches. Pure text-embedding similarity has no concept of
"years of experience", so a senior role that happens to share a lot of
skill vocabulary with a fresher's resume can score just as "similar" as
an actual entry-level posting. This filter corrects for that.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
CHROMA_DIR     = ROOT / "chroma_db"

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "job_postings"

# Module-level singletons (loaded once, reused across calls)
_model:      Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at '{CHROMA_DIR}'. "
                "Run `python src/embed.py` first to build the index."
            )
        client      = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ── Experience estimation & filtering ───────────────────────────────────────────

_YEARS_EXPERIENCE_RE = re.compile(
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
    re.IGNORECASE,
)
_EXPECTED_GRAD_RE = re.compile(r"expected\s+\w+\s+(\d{4})", re.IGNORECASE)
_MIN_YEARS_RE     = re.compile(r"(\d+)")


def estimate_candidate_experience(resume_text: str) -> int:
    """
    Heuristically estimate the candidate's years of professional experience
    from resume text.

    Priority order:
      1. Explicit "X years (of) experience" mentions — use the highest found.
      2. A future/current-year "Expected <Month> <Year>" graduation date, or
         phrases like "fresher" / "final year student" — treated as 0 years.
      3. No clear signal either way — defaults to 0.

    Defaulting to 0 (rather than guessing a mid-level number) is intentional:
    it's safer to under-estimate and show entry-level jobs than to
    over-estimate and surface senior roles to someone who never claimed
    that experience.
    """
    matches = _YEARS_EXPERIENCE_RE.findall(resume_text)
    if matches:
        return max(int(m) for m in matches)

    grad_match = _EXPECTED_GRAD_RE.search(resume_text)
    if grad_match and int(grad_match.group(1)) >= datetime.now().year:
        return 0

    lowered = resume_text.lower()
    if any(kw in lowered for kw in ("fresher", "currently pursuing", "final year student", "final-year student")):
        return 0

    return 0


def _min_years_required(experience_level: str) -> Optional[int]:
    """Parse the minimum years required from strings like '5-10 years'."""
    match = _MIN_YEARS_RE.search(experience_level or "")
    return int(match.group(1)) if match else None


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(
    resume_text: str,
    top_k: int = 5,
    candidate_experience_years: Optional[int] = None,
    experience_buffer: int = 2,
    overfetch_multiplier: int = 30,
) -> list[dict]:
    """
    Embed ``resume_text`` and return the ``top_k`` most similar job postings,
    filtered by experience level so wildly mismatched senior roles don't
    crowd out realistic matches.

    Parameters
    ----------
    resume_text : str
        Plain-text content of the candidate's resume.
    top_k : int
        Number of job postings to return (default: 5).
    candidate_experience_years : int, optional
        Candidate's years of experience. If None, auto-estimated from
        ``resume_text`` via :func:`estimate_candidate_experience`.
    experience_buffer : int
        How many years above the candidate's own experience a job is still
        allowed to require before being filtered out (default: 2). Widened
        automatically if too few results survive filtering at this buffer.
    overfetch_multiplier : int
        How many extra candidates to pull from ChromaDB (top_k * this) before
        filtering, so there's enough left after mismatched jobs are dropped.

    Returns
    -------
    list[dict]
        Each dict has keys: id, title, company, location, experience_level,
        required_skills, document (full embedding text), similarity.
    """
    if not resume_text.strip():
        raise ValueError("resume_text is empty — nothing to retrieve against.")

    if candidate_experience_years is None:
        candidate_experience_years = estimate_candidate_experience(resume_text)

    model      = _get_model()
    collection = _get_collection()

    # Embed the resume
    query_vec = model.encode(resume_text, show_progress_bar=False).tolist()

    # Over-fetch so there's a real pool left after experience filtering
    fetch_n = min(top_k * overfetch_multiplier, collection.count())
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack into candidate dicts (still ordered by similarity from ChromaDB)
    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []
    for job_id, doc, meta, dist in zip(ids, docs, metas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score 0-1: similarity = 1 - dist/2
        similarity = round(1 - dist / 2, 4)
        candidates.append(
            {
                "id":               job_id,
                "title":            meta.get("title", ""),
                "company":          meta.get("company", ""),
                "location":         meta.get("location", ""),
                "experience_level": meta.get("experience_level", ""),
                "required_skills":  meta.get("required_skills", ""),
                "document":         doc,
                "similarity":       similarity,
                "_min_years":       _min_years_required(meta.get("experience_level", "")),
            }
        )

    # Progressively widen the experience buffer until enough candidates survive.
    # Jobs with no parseable experience_level are never filtered out (benefit
    # of the doubt rather than silently dropping them).
    buffer  = experience_buffer
    filtered = candidates
    while buffer <= 20:
        filtered = [
            c for c in candidates
            if c["_min_years"] is None
            or c["_min_years"] <= candidate_experience_years + buffer
        ]
        if len(filtered) >= top_k:
            break
        buffer += 3

    hits = filtered[:top_k]
    for h in hits:
        h.pop("_min_years", None)

    return hits


if __name__ == "__main__":
    # Quick smoke test
    sample_resume = (
        "B.Tech Computer Engineering student, Expected May 2027. "
        "Skills: Python, Flask, scikit-learn, SQL, Gradio, OpenCV. "
        "Projects: Smart Attendance System, Driver Drowsiness Detection, JobLens India. "
        "Data Science internship at Oasis Infobyte."
    )

    est = estimate_candidate_experience(sample_resume)
    print(f"Estimated candidate experience: {est} years\n")

    print("Retrieving top 5 jobs …\n")
    hits = retrieve(sample_resume, top_k=5)
    for i, job in enumerate(hits, 1):
        print(
            f"{i}. {job['title']} — {job['company']}  "
            f"(similarity: {job['similarity']}, requires: {job['experience_level']})"
        )