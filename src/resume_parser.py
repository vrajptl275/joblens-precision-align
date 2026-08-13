"""
resume_parser.py — Extract plain text from a PDF resume (JobLens)

Supports:
  • PDF files via PyMuPDF (fitz)  — primary
  • Plain .txt files              — fallback / testing
  • Raw string passthrough        — for Gradio text-area input
"""

from __future__ import annotations

from pathlib import Path
import re


# ── PDF extraction ─────────────────────────────────────────────────────────────

def _parse_pdf(path: str | Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with:  pip install pymupdf"
        ) from exc

    doc   = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def _parse_txt(path: str | Path) -> str:
    """Read a plain-text file."""
    return Path(path).read_text(encoding="utf-8")


# ── Post-processing ────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """
    Light normalisation:
      - Collapse 3+ blank lines → 2 blank lines
      - Strip trailing whitespace per line
      - Remove null bytes
    """
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_resume(source: str | Path) -> str:
    """
    Parse a resume and return clean plain text.

    Parameters
    ----------
    source : str | Path
        Accepts:
          • Path to a ``.pdf`` file
          • Path to a ``.txt`` file
          • A raw text string (returned as-is after cleaning)

    Returns
    -------
    str
        Clean plain-text resume content.

    Raises
    ------
    FileNotFoundError
        If ``source`` looks like a file path but the file doesn't exist.
    ValueError
        If the extracted text is empty or whitespace-only.
    """
    # Decide whether *source* is a real filesystem path or raw text pasted
    # from the Gradio text area.  Path.exists() itself can raise OSError on
    # macOS/Linux when the string is longer than the OS path limit (~1024
    # chars), so we wrap it in a try/except.
    try:
        path = Path(source) if not isinstance(source, Path) else source
        source_is_file = path.exists() and path.is_file()
        source_is_dir  = path.exists() and not path.is_file()
    except OSError:
        # String too long to be a path — treat as raw text
        source_is_file = False
        source_is_dir  = False

    if source_is_file:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = _parse_pdf(path)
        elif suffix in {".txt", ".md"}:
            text = _parse_txt(path)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Use .pdf or .txt."
            )
    elif source_is_dir:
        raise ValueError(f"'{path}' is a directory, not a file.")
    else:
        # Treat source as a raw text string (e.g., from Gradio text area)
        text = str(source)

    text = _clean(text)

    if not text.strip():
        raise ValueError(
            "Resume text is empty after parsing. "
            "Check the file contents or provide non-empty text."
        )

    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/resume_parser.py <resume.pdf|resume.txt>")
        sys.exit(1)

    result = parse_resume(sys.argv[1])
    print("─" * 60)
    print(result[:2000])
    print("─" * 60)
    print(f"\n[resume_parser] ✅  Extracted {len(result)} characters.")
