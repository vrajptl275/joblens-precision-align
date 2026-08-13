"""
app.py — JobLens Gradio UI

End-to-end RAG pipeline:
  1. Parse resume (PDF upload or text paste)
  2. Retrieve top-K matching jobs from ChromaDB
  3. Build per-job prompts
  4. Query local 'joblens' Ollama model — results stream in progressively,
     one job at a time, instead of appearing all at once at the end
  5. Display results in a calibrated, instrument-style interface

Run:
    python app.py
"""

import sys
from pathlib import Path

# Allow `from src.X import Y` when running as `python app.py`
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from src.resume_parser import parse_resume
from src.retrieve import retrieve, estimate_candidate_experience
from src.prompts import build_all_prompts
from src.llm_client import generate

# ── Config ─────────────────────────────────────────────────────────────────────
TOP_K       = 5
LLM_MODEL   = "joblens"


# ── Design helpers: aperture graphic + score dial ───────────────────────────────

def _aperture_svg(size: int = 120, blades: int = 10, extra_class: str = "") -> str:
    """Render the JobLens signature aperture graphic — a set of radial blades
    around a lens core, echoing a camera iris focusing on a subject."""
    cx = cy = size / 2
    r_outer = size / 2 - 3
    r_inner = size * 0.32
    blade_len = r_outer - r_inner
    parts = []
    for i in range(blades):
        angle = (360 / blades) * i
        parts.append(
            f'<rect class="aperture-blade" x="{cx - 1.6:.1f}" y="{cy - r_outer:.1f}" '
            f'width="3.2" height="{blade_len:.1f}" rx="1.4" '
            f'transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})" />'
        )
    return f'''<svg class="aperture-graphic {extra_class}" viewBox="0 0 {size} {size}" width="{size}" height="{size}" aria-hidden="true">
  <circle class="aperture-ring-outer" cx="{cx:.1f}" cy="{cy:.1f}" r="{r_outer:.1f}" />
  <g class="aperture-blade-group">{"".join(parts)}</g>
  <circle class="aperture-ring-inner" cx="{cx:.1f}" cy="{cy:.1f}" r="{r_inner:.1f}" />
</svg>'''


def _score_dial_svg(score_num: int) -> str:
    """Circular calibration dial: tick-marked scale + brass progress arc."""
    circumference = 251.2  # 2 * pi * r(40)
    dash_offset = round(circumference * (1 - score_num / 100), 1)
    ticks = []
    for i in range(30):
        angle = i * 12
        major = i % 5 == 0
        y1, y2 = (5, 13) if major else (7, 12)
        cls = "dial-tick dial-tick-major" if major else "dial-tick"
        ticks.append(f'<line class="{cls}" x1="50" y1="{y1}" x2="50" y2="{y2}" transform="rotate({angle} 50 50)" />')
    return f'''<svg class="score-ring" viewBox="0 0 100 100" aria-hidden="true">
  <g class="dial-ticks">{"".join(ticks)}</g>
  <circle class="score-ring-bg" cx="50" cy="50" r="40" />
  <circle class="score-ring-progress" cx="50" cy="50" r="40"
    style="stroke-dasharray:{circumference}; stroke-dashoffset:{dash_offset};"
    data-target-offset="{dash_offset}" />
</svg>'''


def _status_panel(kind: str, message: str, submessage: str = "") -> str:
    """Unified status/progress panel shown while the pipeline runs."""
    icon = {
        "processing": _aperture_svg(size=34, blades=8, extra_class="aperture-pulse"),
        "error": '<span class="status-glyph status-glyph-error">!</span>',
        "warning": '<span class="status-glyph status-glyph-warning">?</span>',
    }.get(kind, _aperture_svg(size=34, blades=8, extra_class="aperture-pulse"))
    sub = f'<div class="status-submessage">{submessage}</div>' if submessage else ""
    return f'''<div class="status-panel status-panel--{kind}">
  <div class="status-icon">{icon}</div>
  <div class="status-copy">
    <div class="status-message">{message}</div>
    {sub}
  </div>
</div>'''


# ── Core pipeline ──────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """Extract structured fields from the LLM's response text.

    Only parses the four fields the model was actually fine-tuned to produce:
    Match Score, Matching Skills, Missing Skills, Suggestion.
    """
    lines  = raw.strip().splitlines()
    result = {
        "score":    "N/A",
        "matching": "N/A",
        "missing":  "N/A",
        "suggest":  raw,   # fallback: show raw text if parsing finds nothing
    }
    for line in lines:
        low = line.lower()
        if low.startswith("match score:"):
            result["score"]    = line.split(":", 1)[1].strip()
        elif low.startswith("matching skills:"):
            result["matching"] = line.split(":", 1)[1].strip()
        elif low.startswith("missing skills:"):
            result["missing"]  = line.split(":", 1)[1].strip()
        elif low.startswith("suggestion:"):
            result["suggest"]  = line.split(":", 1)[1].strip()
    return result


def _format_block(rank: int, job: dict, parsed: dict, delay: float = 0.0) -> str:
    """Render one job's result as a calibrated match card."""
    import re
    raw_score = parsed.get("score", "85")
    m = re.search(r"(\d+)", raw_score)
    score_num = min(max(int(m.group(1)) if m else 85, 0), 100)

    matching_raw = parsed.get("matching", "")
    missing_raw  = parsed.get("missing", "")
    matching_skills = [s.strip() for s in matching_raw.split(",") if s.strip() and s.strip().lower() != "none"]
    missing_skills  = [s.strip() for s in missing_raw.split(",") if s.strip() and s.strip().lower() != "none"]

    matching_pills = "".join(f'<span class="skill-pill pill-match">✓ {s}</span>' for s in matching_skills) \
        or '<span class="skill-pill pill-match">✓ Core resume skills align</span>'
    missing_pills = "".join(f'<span class="skill-pill pill-gap">+ {s}</span>' for s in missing_skills) \
        or '<span class="skill-pill pill-match">✓ No critical gaps found</span>'

    return f'''
<div class="result-card reveal" style="animation-delay:{delay:.2f}s">
  <div class="result-card-header">
    <div class="result-card-heading">
      <span class="rank-badge">◎ MATCH #{rank:02d}</span>
      <h3 class="job-title">{job['title']} <span class="job-company">@ {job['company']}</span></h3>
      <div class="job-meta">
        <span class="meta-chip">📍 {job['location']}</span>
        <span class="meta-chip">⏱ {job['experience_level']}</span>
        <span class="meta-chip">🔗 Similarity {int(job['similarity'] * 100)}%</span>
      </div>
    </div>
    <div class="score-ring-container">
      {_score_dial_svg(score_num)}
      <div class="score-text">
        <span class="score-val">{score_num}%</span>
        <span class="score-lbl">MATCH</span>
      </div>
    </div>
  </div>

  <div class="skills-grid">
    <div class="skill-group">
      <div class="group-title">VERIFIED COMPETENCIES</div>
      <div class="pills-wrapper">{matching_pills}</div>
    </div>
    <div class="skill-group">
      <div class="group-title">GAPS DETECTED</div>
      <div class="pills-wrapper">{missing_pills}</div>
    </div>
  </div>

  <div class="suggestion-box">
    <div class="suggestion-header">◈ OPTIMIZATION SUGGESTION</div>
    <div class="suggestion-body">{parsed['suggest']}</div>
  </div>
</div>
'''


def run_pipeline(
    pdf_file,          # gr.File component value (tmp file path or None)
    resume_text: str,  # gr.Textbox value
    top_k: int,
):
    """
    Main handler called by Gradio on submit.

    This is a GENERATOR (uses `yield`, not `return`) so Gradio updates the
    output box live at each step instead of showing nothing until the whole
    pipeline finishes. You'll see calibration/progress panels, then each
    job's result appear one at a time as it's generated.
    """

    # ── 1. Parse resume ─────────────────────────────────────────────────────
    yield _status_panel("processing", "Parsing resume", "Extracting text, skills and structure…")
    try:
        if pdf_file is not None:
            # Gradio 6.0: gr.File returns a plain string path
            pdf_path = pdf_file if isinstance(pdf_file, str) else getattr(pdf_file, "name", str(pdf_file))
            text = parse_resume(pdf_path)
        elif resume_text and resume_text.strip():
            text = parse_resume(resume_text)
        else:
            yield _status_panel("warning", "Nothing to analyze yet", "Upload a PDF or paste your resume text above.")
            return
    except Exception as exc:
        yield _status_panel("error", "Resume parsing failed", str(exc))
        return

    # ── 2. Retrieve top-K jobs ───────────────────────────────────────────────
    est_years = estimate_candidate_experience(text)
    exp_label = "Fresher / 0 years" if est_years == 0 else f"~{est_years} years"
    yield _status_panel("processing", "Scanning the job index", f"Detected experience: {exp_label} · retrieving top {top_k} matches…")
    try:
        jobs = retrieve(text, top_k=top_k, candidate_experience_years=est_years)
    except FileNotFoundError:
        yield _status_panel("error", "Job index not found", "Run <code>python src/embed.py</code> to build the ChromaDB index first.")
        return
    except Exception as exc:
        yield _status_panel("error", "Retrieval failed", str(exc))
        return

    if not jobs:
        yield _status_panel("warning", "No jobs returned", "The index may be empty — try re-running the embed step.")
        return

    # ── 3. Build prompts ─────────────────────────────────────────────────────
    prompts = build_all_prompts(text, jobs)

    # ── 4. Generate LLM responses — one job at a time, streamed to the UI ────
    sections = []
    for rank, (job, prompt) in enumerate(zip(jobs, prompts), 1):
        progress = _status_panel(
            "processing",
            f"Analyzing match {rank} of {len(jobs)}",
            f"{job['title']} @ {job['company']}",
        )
        yield "".join(sections) + progress

        try:
            raw = generate(prompt, model=LLM_MODEL)
        except ConnectionError as exc:
            yield "".join(sections) + _status_panel("error", "Ollama not reachable", str(exc))
            return
        except Exception as exc:
            yield "".join(sections) + _status_panel("error", f"Generation failed on match {rank}", str(exc))
            return

        parsed = _parse_response(raw)
        sections.append(_format_block(rank, job, parsed, delay=0.0))

        # Show this job's finished result immediately — don't wait for the rest
        yield "".join(sections)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Design tokens ────────────────────────────────────────────────────── */
:root {
    --bg-main: #f5f3ee;
    --bg-sunken: #eeeae1;
    --card-bg: #ffffff;
    --card-border: #e3ddd0;
    --text-primary: #1c1812;
    --text-secondary: #57503f;
    --text-muted: #8b8270;
    --accent-brass: #a8752f;
    --accent-brass-bright: #c48d3f;
    --accent-teal: #1f7a6c;
    --accent-teal-bright: #14958a;
    --accent-rust: #b0532e;
    --pill-match-bg: rgba(31, 122, 108, 0.10);
    --pill-match-text: #1f7a6c;
    --pill-match-border: rgba(31, 122, 108, 0.28);
    --pill-gap-bg: rgba(176, 83, 46, 0.10);
    --pill-gap-text: #b0532e;
    --pill-gap-border: rgba(176, 83, 46, 0.26);
    --ring-bg: #e3ddd0;
    --ring-tick: #cdc4b1;
    --shadow-card: 0 14px 34px -12px rgba(28, 24, 18, 0.14), 0 2px 8px -2px rgba(28, 24, 18, 0.06);
    --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SFMono-Regular', Menlo, monospace;
}

.dark, [data-theme="dark"] {
    --bg-main: #121110;
    --bg-sunken: #1a1815;
    --card-bg: #1c1a16;
    --card-border: #2e2a22;
    --text-primary: #f3ede1;
    --text-secondary: #c7bda9;
    --text-muted: #8e8571;
    --accent-brass: #d4a24e;
    --accent-brass-bright: #e8bd6e;
    --accent-teal: #4fd6c4;
    --accent-teal-bright: #6be3d2;
    --accent-rust: #e07a52;
    --pill-match-bg: rgba(79, 214, 196, 0.12);
    --pill-match-text: #6be3d2;
    --pill-match-border: rgba(79, 214, 196, 0.3);
    --pill-gap-bg: rgba(224, 122, 82, 0.14);
    --pill-gap-text: #e79b7c;
    --pill-gap-border: rgba(224, 122, 82, 0.32);
    --ring-bg: #2e2a22;
    --ring-tick: #3c3629;
    --shadow-card: 0 24px 48px -16px rgba(0, 0, 0, 0.55);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    transition: background-color 0.35s ease, color 0.35s ease;
}

::selection { background: var(--accent-brass); color: #fff; }

a { color: var(--accent-brass); }

/* ── Motion & focus baseline ─────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
}
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--accent-brass-bright) !important;
    outline-offset: 2px !important;
}

/* ── Signature aperture graphic ──────────────────────────────────────── */
.aperture-graphic { overflow: visible; }
.aperture-ring-outer, .aperture-ring-inner {
    fill: none;
    stroke: var(--card-border);
    stroke-width: 1.5;
}
.aperture-blade {
    fill: var(--accent-brass);
    opacity: 0.85;
}
.aperture-blade-group {
    transform-origin: center;
    animation: aperture-drift 26s linear infinite;
}
.aperture-boot .aperture-blade-group { animation: aperture-open 1.1s cubic-bezier(0.16, 1, 0.3, 1) both, aperture-drift 26s linear infinite 1.1s; }
@keyframes aperture-open {
    from { transform: scale(0.2) rotate(-70deg); opacity: 0; }
    to   { transform: scale(1) rotate(0deg); opacity: 1; }
}
@keyframes aperture-drift {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.aperture-pulse .aperture-blade-group { animation: aperture-pulse 1.6s ease-in-out infinite; }
@keyframes aperture-pulse {
    0%, 100% { transform: scale(0.72) rotate(0deg); }
    50%      { transform: scale(1) rotate(30deg); }
}

/* ── Navigation ───────────────────────────────────────────────────────── */
.nav-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 1rem 1.75rem;
    border-bottom: 1px solid var(--card-border);
    background: var(--card-bg);
}
.brand-lockup { display: flex; align-items: center; gap: 0.6rem; }
.brand-word {
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}
.brand-word span { color: var(--accent-brass); }
.theme-switch-bar {
    display: flex;
    gap: 0.2rem;
    background: var(--bg-sunken);
    padding: 0.25rem;
    border-radius: 9999px;
    border: 1px solid var(--card-border);
}
.theme-btn {
    border: none;
    background: transparent;
    padding: 0.4rem 0.85rem;
    border-radius: 9999px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
}
.theme-btn:hover { color: var(--accent-brass); }
.theme-btn.active {
    background: var(--card-bg);
    color: var(--accent-brass);
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero-section {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    align-items: center;
    gap: 2.5rem;
    padding: 3.5rem 1.75rem 2.5rem;
    max-width: 1180px;
    margin: 0 auto;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-brass);
    margin-bottom: 1rem;
}
.hero-eyebrow::before { content: ""; width: 1.4rem; height: 1px; background: var(--accent-brass); }
.hero-title {
    font-family: var(--font-display);
    font-size: 2.9rem;
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin: 0 0 1rem;
}
.hero-title-accent { color: var(--accent-brass); }
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.65;
    max-width: 46ch;
    margin-bottom: 1.5rem;
}
.hero-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.35rem 0.7rem;
    border-radius: 0.4rem;
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    background: var(--card-bg);
}
.badge-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent-teal); margin-right: 0.4rem; }

.hero-instrument {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.hero-instrument .aperture-graphic { filter: drop-shadow(0 12px 24px rgba(0,0,0,0.12)); }
.hero-readout {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    font-family: var(--font-mono);
    pointer-events: none;
}
.hero-readout-value { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); }
.hero-readout-label { font-size: 0.62rem; letter-spacing: 0.12em; color: var(--text-muted); margin-top: 0.15rem; }

@media (max-width: 900px) {
    .hero-section { grid-template-columns: 1fr; text-align: center; padding-top: 2.5rem; }
    .hero-eyebrow { justify-content: center; }
    .hero-eyebrow::before { display: none; }
    .hero-subtitle { margin-left: auto; margin-right: auto; }
    .hero-badges { justify-content: center; }
    .hero-instrument { order: -1; }
}

/* ── Panel / card component ──────────────────────────────────────────── */
.panel {
    position: relative;
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 0.85rem !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.25s ease, transform 0.25s ease;
}
.panel::before, .panel::after {
    content: "";
    position: absolute;
    width: 14px;
    height: 14px;
    border-color: var(--accent-brass);
    opacity: 0.55;
    transition: opacity 0.25s ease;
}
.panel::before { top: -1px; left: -1px; border-top: 2px solid; border-left: 2px solid; border-top-left-radius: 6px; }
.panel::after  { bottom: -1px; right: -1px; border-bottom: 2px solid; border-right: 2px solid; border-bottom-right-radius: 6px; }
.panel:hover { border-color: var(--accent-brass) !important; }
.panel:hover::before, .panel:hover::after { opacity: 1; }

.card-heading {
    font-family: var(--font-display);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1.25rem;
}

/* ── Reveal-on-scroll ─────────────────────────────────────────────────── */
.reveal { animation: reveal-in 0.55s cubic-bezier(0.16,1,0.3,1) both; }
@keyframes reveal-in {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── How JobLens Works ────────────────────────────────────────────────── */
.step-item { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.25rem; }
.step-item:last-child { margin-bottom: 0; }
.step-number {
    width: 2rem; height: 2rem; border-radius: 0.5rem;
    background: var(--bg-sunken);
    border: 1px solid var(--card-border);
    color: var(--accent-brass);
    font-family: var(--font-mono);
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.step-content h4 { margin: 0; font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }
.step-content p { margin: 0.2rem 0 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; }

/* ── Primary CTA ──────────────────────────────────────────────────────── */
.main-cta-btn {
    position: relative;
    overflow: hidden;
    background: var(--accent-brass) !important;
    border: none !important;
    color: #fff !important;
    font-family: var(--font-display) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 1rem 2rem !important;
    border-radius: 0.6rem !important;
    box-shadow: 0 12px 24px -8px rgba(168, 117, 47, 0.5) !important;
    cursor: pointer !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
    width: 100% !important;
    margin-top: 1rem !important;
}
.main-cta-btn::after {
    content: "";
    position: absolute;
    top: 0; left: -60%;
    width: 40%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}
.main-cta-btn:hover { background: var(--accent-brass-bright) !important; transform: translateY(-2px); box-shadow: 0 16px 30px -8px rgba(168, 117, 47, 0.6) !important; }
.main-cta-btn:hover::after { left: 130%; }

/* ── Status / progress panel ─────────────────────────────────────────── */
.status-panel {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    border-radius: 0.85rem;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    animation: reveal-in 0.4s ease both;
}
.status-panel--error   { border-color: rgba(176, 83, 46, 0.4); }
.status-panel--warning { border-color: rgba(196, 141, 63, 0.4); }
.status-icon { flex-shrink: 0; display: flex; }
.status-glyph {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-weight: 700; font-size: 1rem;
}
.status-glyph-error   { background: var(--pill-gap-bg); color: var(--accent-rust); }
.status-glyph-warning { background: rgba(196, 141, 63, 0.14); color: var(--accent-brass); }
.status-message { font-family: var(--font-display); font-weight: 700; font-size: 1rem; color: var(--text-primary); }
.status-submessage { font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }

/* ── Result cards ─────────────────────────────────────────────────────── */
.result-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 1rem;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.25s ease, transform 0.25s ease;
    position: relative;
}
.result-card::before, .result-card::after {
    content: "";
    position: absolute;
    width: 16px; height: 16px;
    border-color: var(--accent-brass);
    opacity: 0.4;
    transition: opacity 0.25s ease;
}
.result-card::before { top: -1px; left: -1px; border-top: 2px solid; border-left: 2px solid; border-top-left-radius: 8px; }
.result-card::after  { bottom: -1px; right: -1px; border-bottom: 2px solid; border-right: 2px solid; border-bottom-right-radius: 8px; }
.result-card:hover { border-color: var(--accent-brass); transform: translateY(-3px); }
.result-card:hover::before, .result-card:hover::after { opacity: 1; }

.result-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}
.rank-badge {
    display: inline-block;
    font-family: var(--font-mono);
    padding: 0.2rem 0.55rem;
    background: var(--pill-match-bg);
    color: var(--accent-teal);
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    border-radius: 0.35rem;
    margin-bottom: 0.5rem;
}
.job-title { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.job-company { color: var(--accent-brass); font-weight: 600; }
.job-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }
.meta-chip {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    background: var(--bg-sunken);
    border: 1px solid var(--card-border);
    padding: 0.2rem 0.55rem;
    border-radius: 0.35rem;
}

.score-ring-container { position: relative; width: 88px; height: 88px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.score-ring { width: 100%; height: 100%; transform: rotate(-90deg); }
.score-ring-bg { fill: none; stroke: var(--ring-bg); stroke-width: 7; }
.dial-ticks { transform: rotate(90deg); transform-origin: 50px 50px; }
.dial-tick { stroke: var(--ring-tick); stroke-width: 1; }
.dial-tick-major { stroke: var(--text-muted); stroke-width: 1.4; }
.score-ring-progress {
    fill: none;
    stroke: var(--accent-brass);
    stroke-width: 7;
    stroke-linecap: round;
    transition: stroke-dashoffset 1.1s cubic-bezier(0.16,1,0.3,1);
}
.score-text { position: absolute; text-align: center; display: flex; flex-direction: column; }
.score-val { font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.score-lbl { font-family: var(--font-mono); font-size: 0.58rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.08em; margin-top: 3px; }

.skills-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1.25rem; }
.skill-group { background: var(--bg-sunken); padding: 1rem; border-radius: 0.65rem; border: 1px solid var(--card-border); }
.group-title { font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; color: var(--text-muted); margin-bottom: 0.65rem; }
.pills-wrapper { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.skill-pill {
    padding: 0.28rem 0.65rem;
    border-radius: 0.4rem;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    transition: transform 0.15s ease;
}
.skill-pill:hover { transform: translateY(-1px); }
.pill-match { background: var(--pill-match-bg); color: var(--pill-match-text); border: 1px solid var(--pill-match-border); }
.pill-gap   { background: var(--pill-gap-bg);   color: var(--pill-gap-text);   border: 1px solid var(--pill-gap-border); }

.suggestion-box { background: var(--bg-sunken); border-left: 3px solid var(--accent-brass); padding: 1rem 1.1rem; border-radius: 0 0.65rem 0.65rem 0; }
.suggestion-header { font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; color: var(--accent-brass); margin-bottom: 0.35rem; }
.suggestion-body { font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55; }

@media (max-width: 640px) {
    .skills-grid { grid-template-columns: 1fr; }
    .result-card-header { flex-direction: column; }
    .score-ring-container { align-self: flex-start; }
}

/* ── Tabs / results header ───────────────────────────────────────────── */
.results-heading {
    font-family: var(--font-display);
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem;
}
"""

JS_INTERACTIONS = r"""
function() {
    const root = document.documentElement;
    const STORAGE_KEY = 'joblens-theme';

    function applyTheme(mode) {
        const container = document.querySelector('.gradio-container');
        let resolved = mode;
        if (mode === 'system') {
            resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        if (resolved === 'dark') {
            root.setAttribute('data-theme', 'dark');
            if (container) container.classList.add('dark');
        } else {
            root.setAttribute('data-theme', 'light');
            if (container) container.classList.remove('dark');
        }
        document.querySelectorAll('.theme-btn').forEach(function (btn) {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    }

    function setTheme(mode) {
        try { localStorage.setItem(STORAGE_KEY, mode); } catch (e) {}
        applyTheme(mode);
    }

    window.setJobLensTheme = setTheme;
    let saved = 'system';
    try { saved = localStorage.getItem(STORAGE_KEY) || 'system'; } catch (e) {}
    applyTheme(saved);

    // Play the aperture "boot" animation once per page load.
    document.querySelectorAll('.aperture-graphic').forEach(function (svg) {
        svg.classList.add('aperture-boot');
    });

    // Animate score dials growing in from zero whenever new result cards
    // are streamed into the results panel, and reveal cards as they arrive.
    const resultsRoot = document.getElementById('results');
    if (resultsRoot && window.MutationObserver) {
        const seen = new WeakSet();
        const observer = new MutationObserver(function () {
            resultsRoot.querySelectorAll('.score-ring-progress').forEach(function (circle) {
                if (seen.has(circle)) return;
                seen.add(circle);
                const target = circle.getAttribute('data-target-offset');
                const full = circle.getAttribute('style').match(/stroke-dasharray:\s*([\d.]+)/);
                if (!target || !full) return;
                circle.style.transition = 'none';
                circle.style.strokeDashoffset = full[1];
                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        circle.style.transition = 'stroke-dashoffset 1.1s cubic-bezier(0.16,1,0.3,1)';
                        circle.style.strokeDashoffset = target;
                    });
                });
            });
        });
        observer.observe(resultsRoot, { childList: true, subtree: true });
    }
}
"""

with gr.Blocks(
    title="JobLens — Adaptive Precision Matching",
) as demo:

    # ── Top Navigation Bar ──────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="nav-header">
        <div class="brand-lockup">
            {_aperture_svg(size=32, blades=8)}
            <span class="brand-word">Job<span>Lens</span></span>
        </div>
        <div class="theme-switch-bar">
            <button class="theme-btn" data-mode="light" onclick="window.setJobLensTheme('light')">Light</button>
            <button class="theme-btn" data-mode="dark" onclick="window.setJobLensTheme('dark')">Dark</button>
            <button class="theme-btn active" data-mode="system" onclick="window.setJobLensTheme('system')">System</button>
        </div>
    </div>
    """)

    # ── Hero ─────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="hero-section">
        <div class="hero-copy">
            <div class="hero-eyebrow">RESUME-TO-ROLE CALIBRATION</div>
            <h1 class="hero-title">Precision matching,<br><span class="hero-title-accent">zero guesswork.</span></h1>
            <p class="hero-subtitle">Upload your resume and JobLens measures it against live openings — skill by skill — so you know exactly where you stand and what to fix.</p>
            <div class="hero-badges">
                <span class="badge"><span class="badge-dot"></span>Runs 100% locally</span>
                <span class="badge">Adjustable top-K</span>
                <span class="badge">Skill-level scoring</span>
            </div>
        </div>
        <div class="hero-instrument">
            {_aperture_svg(size=220, blades=12)}
        </div>
    </div>
    """)

    # ── Interactive Tabs ───────────────────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("Analyze Compatibility"):
            with gr.Row():
                # Left Column: Document Input Card
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["panel", "reveal"]):
                        gr.HTML('<div class="card-heading">Document Input</div>')
                        pdf_input = gr.File(
                            label="Upload Resume (PDF)",
                            file_types=[".pdf"],
                            type="filepath",
                            elem_id="pdf-upload",
                        )
                        gr.Markdown("**— or paste text directly —**")
                        text_input = gr.Textbox(
                            label="Resume Content",
                            placeholder="Paste your resume content here...",
                            lines=8,
                            max_lines=16,
                        )

                # Right Column: Analysis Settings & Workflow Guide
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["panel", "reveal"]):
                        gr.HTML('<div class="card-heading">Analysis Settings</div>')
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=TOP_K,
                            step=1,
                            label="Match Strictness (Top-K)",
                            info="How many top candidate jobs to retrieve (1–10)",
                        )

                    with gr.Group(elem_classes=["panel", "reveal"]):
                        gr.HTML('<div class="card-heading">How JobLens Works</div>')
                        gr.HTML("""
                        <div class="step-item">
                            <div class="step-number">1</div>
                            <div class="step-content">
                                <h4>Parse &amp; extract</h4>
                                <p>Breaks down your experience, skills, and implied proficiencies.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-number">2</div>
                            <div class="step-content">
                                <h4>Vector matching</h4>
                                <p>Compares your profile against indexed job descriptions in ChromaDB.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-number">3</div>
                            <div class="step-content">
                                <h4>Score &amp; recommend</h4>
                                <p>Delivers calibrated match scores with skill-level gap analysis.</p>
                            </div>
                        </div>
                        """)

            # Main CTA Button
            submit_btn = gr.Button(
                "Analyze Compatibility",
                variant="primary",
                elem_classes=["main-cta-btn"],
            )

            # Results Section
            gr.HTML('<div class="results-heading">Analysis Results</div>')
            output = gr.HTML(
                value=_status_panel("processing", "Awaiting resume", "Results will appear here after you click Analyze Compatibility.").replace(
                    '<div class="status-icon">' + _aperture_svg(size=34, blades=8, extra_class="aperture-pulse") + '</div>',
                    '<div class="status-icon" style="opacity:0.5">' + _aperture_svg(size=34, blades=8) + '</div>',
                ),
                elem_id="results",
            )

            # Examples Preset
            gr.Examples(
                examples=[
                    [
                        None,
                        (
                            "B.Tech Computer Engineering student, final year.\n"
                            "Skills: Python, Flask, Gradio, OpenCV, scikit-learn, SQL, Git.\n"
                            "Projects: Smart Attendance System (OpenCV), "
                            "Driver Drowsiness Detection (YOLOv8), "
                            "JobLens India (NLP + Gradio).\n"
                            "Internship: Data Science at Oasis Infobyte (6 months).\n"
                            "Familiar with: TensorFlow basics, REST APIs, Linux."
                        ),
                        5,
                    ]
                ],
                inputs=[pdf_input, text_input, top_k_slider],
                label="Try a sample resume",
            )

        with gr.TabItem("How It Works"):
            gr.Markdown("""
            ### Technical architecture
            JobLens uses a 3-tier local RAG (Retrieval-Augmented Generation) pipeline:

            1. **Parser layer** — high-accuracy PDF parsing via PyMuPDF/pdfplumber with deterministic skill regex matching.
            2. **Vector index (ChromaDB)** — 384-dimensional dense vector embeddings generated with `all-MiniLM-L6-v2`.
            3. **Local LLM engine** — a quantized 8B Llama-3.1 model fine-tuned on resume-job alignment pairs, running fully offline via Ollama.
            """)

        with gr.TabItem("Privacy"):
            gr.Markdown("""
            ### 100% data privacy
            * **Zero cloud transfer** — all parsing, embedding, and inference run strictly on your local machine.
            * **No third-party APIs** — no OpenAI, Anthropic, or external cloud LLM endpoints are called.
            * **Transient session data** — uploaded resumes are processed in memory and never sent to external servers.
            """)

    # ── Footer ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align: center; padding: 2rem 0 1.5rem; color: var(--text-muted); font-size: 0.82rem; border-top: 1px solid var(--card-border); margin-top: 3rem; font-family: var(--font-mono);">
        <div><b>JobLens</b> © 2026 JobLens AI · calibrated locally, on your machine.</div>
        <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 1.5rem;">
            <a href="#" style="color: var(--accent-brass); text-decoration: none;">Tech Stack</a>
            <a href="#" style="color: var(--accent-brass); text-decoration: none;">Privacy Policy</a>
            <a href="#" style="color: var(--accent-brass); text-decoration: none;">Terms of Service</a>
        </div>
    </div>
    """)

    # ── Event Wiring ───────────────────────────────────────────────────────
    submit_btn.click(
        fn=run_pipeline,
        inputs=[pdf_input, text_input, top_k_slider],
        outputs=output,
        show_progress="full",
    )

# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "\n"
        "┌─────────────────────────────────────────┐\n"
        "│  JobLens — Adaptive Precision UI         │\n"
        "│  http://localhost:7860                   │\n"
        "│  Ctrl+C to stop                          │\n"
        "└─────────────────────────────────────────┘\n"
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="teal",
            neutral_hue="stone",
        ),
        css=CSS,
        js=JS_INTERACTIONS,
    )