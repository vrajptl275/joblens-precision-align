# JobLens — Resume-to-Job Matching with Local RAG

JobLens is a **fully offline**, privacy-first Resume-to-Job matching system built on a local RAG (Retrieval-Augmented Generation) pipeline. Upload your resume and get calibrated match scores, skill gap analysis, and actionable suggestions — all processed entirely on your machine.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![Ollama](https://img.shields.io/badge/LLM-Ollama-green)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-purple)

---

## 1. Project Overview

Matching a resume against many job postings isn't a simple keyword search — meaning matters more than exact wording (e.g., "built REST APIs with Flask" should match "backend API development experience" even without shared keywords). 

This project solves this by combining **semantic retrieval** (finding the most relevant postings for a resume) with a **local LLM** that reads only those relevant postings and writes a match score, skill-gap analysis, and suggestion — entirely offline, with no cloud API dependencies.

### The Core Problem RAG Solves
With thousands of job postings, you cannot paste all of them into one LLM context window — it gets slow, expensive, and dilutes attention. Retrieval acts as a math-based router that filters down the corpus to the $K$ most relevant roles before the LLM evaluates the candidates.

---

## 2. Plain-Language System Walkthrough

Here is exactly what happens when you click **Analyze Compatibility**:

1. **Text Extraction:** The system reads the plain text from your PDF resume (using PyMuPDF).
2. **Resume Fingerprinting (Embedding):** A lightweight embedding model (`all-MiniLM-L6-v2`) converts your resume's semantic meaning into a 384-dimensional number string (fingerprint). 
3. **Pre-Indexed Job Corpus:** Every job posting in the database has already been fingerprinted ahead of time and stored in ChromaDB.
4. **Vector Search:** The system mathematically compares your resume's fingerprint against the pre-indexed jobs, identifying candidate listings with matching context.
5. **Experience Filter:** Candidate experience is estimated from the resume. Mismatched jobs (e.g. senior engineer positions for a fresher) are pruned out using a progressive buffer search.
6. **Prompt Assembly:** For each matching job, a structured prompt is built containing the resume, the job details, and pre-computed matching/missing skills.
7. **LLM Evaluation:** The local fine-tuned Ollama model parses the prompt and generates a match score and actionable resume optimization suggestions.

---

## 3. System Architecture & Tech Stack

```
                    ┌───────────────────────┐
                    │   Job Postings Data     │  (JSON, 3,000 India job postings)
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Embedding Pipeline     │  sentence-transformers
                    │  (fingerprint each job) │  (all-MiniLM-L6-v2)
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Vector Store          │  ChromaDB (cosine metric space)
                    │  (stored job fingerprints)│
                    └───────────┬─────────────┘
                                │
   Resume PDF ──► Parse text ──► Fingerprint resume ──► Filter by Experience &
     (PyMuPDF)   (resume_parser) (all-MiniLM-L6-v2)     Similarity (retrieve.py)
                                                                 │
                                                                 ▼
                                                  ┌───────────────────────┐
                                                  │  Skill Matcher        │
                                                  │  (Synonym mappings &  │
                                                  │   content filters)    │
                                                  └───────────┬───────────┘
                                                                 │
                                                                 ▼
                                                  ┌───────────────────────┐
                                                  │  Prompt Builder         │
                                                  │  (Assembles input for  │
                                                  │   fine-tuned LLM format)│
                                                  └───────────┬───────────┘
                                                                 │
                                                                 ▼
                                                  ┌───────────────────────┐
                                                  │  Local LLM              │
                                                  │  (Ollama — joblens)     │
                                                  └───────────┬───────────┘
                                                                 │
                                                                 ▼
                                                        Gradio Web UI
                                                 (Streams matches progressively)
```

### What uses AI vs. what is pure math/logic

| Stage | Process Type | Technology Used |
|---|---|---|
| **Resume Parsing** | Pure logic / Python regex | `PyMuPDF (fitz)` + custom sanitization |
| **Resume Fingerprinting** | Neural representation | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Candidate Retrieval** | Dense vector search | Cosine distance metric in `ChromaDB` |
| **Experience Check** | Non-AI regex heuristic | `estimate_candidate_experience` with progressive buffer |
| **Skill Intersection** | Synonym-aware logic | `skill_matcher.py` (resolves synonyms like "github" → "git") |
| **Match Evaluation** | Generative AI | Ollama running the fine-tuned `joblens` Llama-3.1 model |

---

## 4. Prompt & Fine-Tuning Specification

### The Inference Prompt
This prompt is constructed at runtime for each retrieved job. It mirrors the exact token structure the model was fine-tuned on:

```text
Resume: [Candidate Resume Text]
Job: [Job Title, Company, Location, Experience, Description]

Skill analysis (already computed — use exactly as written):
Matching Skills: [Pre-computed matches]
Missing Skills: [Pre-computed gaps]

Respond in this EXACT format:
Match Score: <number>/100
Matching Skills: [Pre-computed matches]
Missing Skills: [Pre-computed gaps]
Suggestion: <one actionable sentence>
```

### LoRA Fine-Tuning
The model was fine-tuned using a custom LoRA adapter (trained using **Unsloth** / Hugging Face `peft` libraries). The dataset consists of positive, negative, and intermediate matching pairs to teach the model how to grade candidate suitability objectively:

```json
{"input": "Resume: B.Tech CE student. Skills: Python, Flask, Gradio, scikit-learn, NLP basics.\nJob: Python Developer at DataSoft. Requires Python, Flask, PostgreSQL, REST API. 0-1 years.", "output": "Match Score: 79/100\nMatching Skills: Python, Flask, REST API\nMissing Skills: PostgreSQL\nSuggestion: Mention specific database projects in your resume, or note your willingness to quickly adapt to PostgreSQL."}
```

---

## 5. Project Structure

```
joblens/
├── app.py                  # Gradio UI + RAG pipeline orchestration
├── requirements.txt        # Python dependencies
├── README.md               # Unified project documentation
├── src/
│   ├── __init__.py
│   ├── embed.py            # Build ChromaDB index from jobs.json
│   ├── resume_parser.py    # PDF/text resume extraction (Gradio-safe)
│   ├── retrieve.py         # Cosine similarity retrieval + experience filtering
│   ├── skill_matcher.py    # Synonym-aware skill matching & keyword normalization
│   ├── prompts.py          # Structured prompt builder matching LoRA formats
│   └── llm_client.py       # Local Ollama HTTP API adapter
├── data/
│   ├── jobs.json           # 3,000 job postings dataset (India)
│   ├── job.json            # Compact schema representation
│   ├── resume.json         # Sample candidate resume profile
│   └── training_data.jsonl # Labeled training dataset for LoRA alignment
├── model/
│   ├── Modelfile           # Ollama model definition with parameters
│   └── resume_matcher_q4_k_m.gguf  # Quantized weights (4.6 GB, not in git)
└── lora_model/             # Fine-tuned adapter parameters (not in git)
```

---

## 6. Setup & Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- At least 8 GB RAM (16 GB recommended for smooth inference)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/joblens.git
   cd joblens
   ```

2. **Create and activate the environment:**
   ```bash
   conda create -n joblens python=3.11 -y
   conda activate joblens
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Index the job dataset:**
   Build the ChromaDB vector index from the 3,000 job corpus (runs completely locally):
   ```bash
   python src/embed.py
   ```

5. **Load the Ollama model:**
   Make sure you have placed the GGUF file `resume_matcher_q4_k_m.gguf` inside the `model/` directory, then register it in Ollama:
   ```bash
   ollama create joblens -f model/Modelfile
   ```

6. **Start the Web Interface:**
   ```bash
   python app.py
   ```
   Open **http://localhost:7860** in your browser.

---

## 7. Large Files Exclusion

Due to GitHub's 100 MB file limit, the following model weights are excluded from the repository. They are kept locally on your target machine:

| File | Size | Description |
|:---|:---|:---|
| `model/resume_matcher_q4_k_m.gguf` | 4.6 GB | The quantized base LLM fine-tuned for candidate scoring. |
| `lora_model/adapter_model.safetensors` | 160 MB | LoRA fine-tuning adapter weights. |
| `lora_model/tokenizer.json` | 16 MB | The base model token mappings. |

---

## 8. Privacy

- **100% Local Inference** — Uploaded resumes are processed in-memory. They are never sent to external servers or cloud endpoints.
- **No Third-Party APIs** — No dependence on OpenAI, Anthropic, or external API keys.

---

## 9. License

This project is licensed under the MIT License - see the LICENSE file for details.
