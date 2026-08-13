"""
embed.py — Embedding + indexing pipeline for JobLens
Reads jobs.json, generates sentence-transformer embeddings, and stores
them in a persistent ChromaDB collection.

Run once before the app:
    python src/embed.py
"""

import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
JOBS_PATH   = ROOT / "data" / "jobs.json"
CHROMA_DIR  = ROOT / "chroma_db"

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"   # fast, 384-dim, runs fully offline
COLLECTION_NAME = "job_postings"
BATCH_SIZE     = 64                    # adjust if you hit OOM


def _build_doc(job: dict) -> str:
    """Combine title + description + skills into one embeddable string."""
    skills = ", ".join(job.get("required_skills", []))
    return (
        f"Title: {job.get('title', '')}\n"
        f"Company: {job.get('company', '')}\n"
        f"Location: {job.get('location', '')}\n"
        f"Skills: {skills}\n"
        f"Experience: {job.get('experience_level', '')}\n"
        f"Description: {job.get('description', '')}"
    )


def build_index(force: bool = False) -> chromadb.Collection:
    """
    Load jobs.json, embed every posting, and upsert into ChromaDB.

    Parameters
    ----------
    force : bool
        If True, delete and rebuild the collection from scratch.

    Returns
    -------
    chromadb.Collection
        The populated ChromaDB collection.
    """
    # ── Load jobs ─────────────────────────────────────────────────────────────
    print(f"[embed] Loading jobs from {JOBS_PATH} …")
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        jobs: list[dict] = json.load(f)
    print(f"[embed] {len(jobs)} job postings loaded.")

    # ── ChromaDB client ───────────────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[embed] Existing collection '{COLLECTION_NAME}' deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    # Skip if already fully indexed
    existing = collection.count()
    if existing >= len(jobs) and not force:
        print(f"[embed] Collection already has {existing} docs — skipping re-embed.")
        return collection

    # ── Embedding model ───────────────────────────────────────────────────────
    print(f"[embed] Loading embedding model '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    # ── Batch upsert ──────────────────────────────────────────────────────────
    print(f"[embed] Embedding {len(jobs)} jobs in batches of {BATCH_SIZE} …")
    for i in tqdm(range(0, len(jobs), BATCH_SIZE), unit="batch"):
        batch = jobs[i : i + BATCH_SIZE]

        ids        = [str(j.get("id", f"job_{i+k}")) for k, j in enumerate(batch)]
        documents  = [_build_doc(j) for j in batch]
        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        metadatas  = [
            {
                "title":            j.get("title", ""),
                "company":          j.get("company", ""),
                "location":         j.get("location", ""),
                "experience_level": j.get("experience_level", ""),
                "required_skills":  ", ".join(j.get("required_skills", [])),
            }
            for j in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"[embed] ✅ Done — {collection.count()} docs in '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB index from jobs.json")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    args = parser.parse_args()

    build_index(force=args.force)
