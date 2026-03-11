"""pdf_parser.py — Sentinel @pdf: attachment handler.

Extracts text from a local PDF file and returns a structured attachment
dict.  Uses the following extraction strategy (in priority order):

  1. pdfplumber — if installed, provides accurate text with layout
  2. pypdf2 / PyPDF2 — fallback pure-Python extractor
  3. Raw byte scan — last resort: strips binary noise and collects
     printable ASCII runs when neither library is available

The extracted text is cleaned and de-hyphenated across page breaks.
Page boundaries are preserved as ``── Page N ──`` markers.

Returns up to ``max_pages`` pages (default all) and supports a page
range via ``start_page`` / ``end_page`` (1-based, inclusive).
"""

import importlib
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Maximum characters to return per attachment (≈ 200 000 chars / ~50 000 tokens)
_MAX_CHARS = 200_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(
    path: str,
    project_root: str = "",
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_chars: int = _MAX_CHARS,
) -> Dict[str, Any]:
    """Extract text from a PDF and return a structured attachment dict.

    Args:
        path: Absolute or project-relative path to the PDF file.
        project_root: Optional project root for relative path resolution.
        start_page: First page to extract (1-based, inclusive).
        end_page: Last page to extract (1-based, inclusive).  None = all.
        max_chars: Maximum characters in the returned content.

    Returns:
        Attachment dict with keys:
            - ``type``        — always ``"pdf"``
            - ``path``        — resolved absolute path
            - ``relative_path`` — path relative to project root
            - ``page_count``   — total number of pages in the document
            - ``extracted_pages`` — number of pages actually extracted
            - ``content``     — extracted plain text
            - ``encoding``    — always ``"text"``
            - ``size_bytes``  — file size in bytes
            - ``truncated``   — True if content was cut at max_chars
            - ``extractor``   — name of back-end used (``"pdfplumber"``,
                                ``"pypdf2"``, or ``"raw"``)

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a PDF (wrong magic bytes).
    """
    resolved = _resolve(path, project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"PDF not found: {resolved}")

    _assert_pdf(resolved)

    raw_bytes = resolved.read_bytes()
    size = len(raw_bytes)
    rel = _relative(resolved, project_root)

    extractor_name, page_count, pages_text = _extract(raw_bytes, start_page, end_page)

    content = _assemble(pages_text)
    truncated = len(content) > max_chars
    content = content[:max_chars]

    return {
        "type": "pdf",
        "path": str(resolved),
        "relative_path": rel,
        "page_count": page_count,
        "extracted_pages": len(pages_text),
        "content": content,
        "encoding": "text",
        "size_bytes": size,
        "truncated": truncated,
        "extractor": extractor_name,
    }


# ---------------------------------------------------------------------------
# Extraction back-ends
# ---------------------------------------------------------------------------


def _extract(
    raw_bytes: bytes,
    start_page: int,
    end_page: Optional[int],
) -> Tuple[str, int, List[str]]:
    """Try each extraction back-end in priority order.

    Args:
        raw_bytes: Full PDF file bytes.
        start_page: First page to extract (1-based).
        end_page: Last page to extract (1-based).  None = all.

    Returns:
        Tuple of (extractor_name, total_page_count, list_of_page_texts).
    """
    # 1. pdfplumber
    pdfplumber = _try_import("pdfplumber")
    if pdfplumber is not None:
        try:
            return _extract_pdfplumber(pdfplumber, raw_bytes, start_page, end_page)
        except Exception:
            pass

    # 2. pypdf2 / PyPDF2
    pypdf2 = _try_import("pypdf2") or _try_import("PyPDF2")
    if pypdf2 is not None:
        try:
            return _extract_pypdf2(pypdf2, raw_bytes, start_page, end_page)
        except Exception:
            pass

    # 3. Raw scan
    return _extract_raw(raw_bytes, start_page, end_page)


def _extract_pdfplumber(
    pdfplumber: Any,
    raw_bytes: bytes,
    start_page: int,
    end_page: Optional[int],
) -> Tuple[str, int, List[str]]:
    """Extract text using pdfplumber.

    Args:
        pdfplumber: The imported pdfplumber module.
        raw_bytes: PDF bytes.
        start_page: First page (1-based).
        end_page: Last page (1-based) or None.

    Returns:
        (extractor_name, total_pages, page_texts)
    """
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        total = len(pdf.pages)
        sliced = _slice_pages(pdf.pages, start_page, end_page, total)
        texts: List[str] = []
        for page in sliced:
            text = page.extract_text() or ""
            texts.append(_clean(text))
    return "pdfplumber", total, texts


def _extract_pypdf2(
    pypdf2: Any,
    raw_bytes: bytes,
    start_page: int,
    end_page: Optional[int],
) -> Tuple[str, int, List[str]]:
    """Extract text using PyPDF2.

    Args:
        pypdf2: The imported pypdf2 / PyPDF2 module.
        raw_bytes: PDF bytes.
        start_page: First page (1-based).
        end_page: Last page (1-based) or None.

    Returns:
        (extractor_name, total_pages, page_texts)
    """
    reader_cls = getattr(pypdf2, "PdfReader", None) or getattr(pypdf2, "PdfFileReader", None)
    reader = reader_cls(io.BytesIO(raw_bytes))
    total = len(reader.pages)
    sliced = _slice_pages(reader.pages, start_page, end_page, total)
    texts: List[str] = []
    for page in sliced:
        text = page.extract_text() or ""
        texts.append(_clean(text))
    return "pypdf2", total, texts


def _extract_raw(
    raw_bytes: bytes,
    start_page: int,
    end_page: Optional[int],
) -> Tuple[str, int, List[str]]:
    """Last-resort extraction: gather printable ASCII runs from raw bytes.

    Splits on ``/Page`` markers as a crude page boundary.

    Args:
        raw_bytes: PDF bytes.
        start_page: First page (1-based).
        end_page: Last page (1-based) or None.

    Returns:
        (extractor_name, total_pages, page_texts)
    """
    # Split on page object markers
    chunks = re.split(rb"/Page\b", raw_bytes)
    total = max(1, len(chunks) - 1)

    start_idx = max(0, start_page - 1)
    end_idx = (end_page or total)
    slices = chunks[start_idx: end_idx]

    texts: List[str] = []
    for chunk in slices:
        # Extract printable ASCII
        printable = re.findall(rb"[ -~]{4,}", chunk)
        text = " ".join(p.decode("ascii", errors="ignore") for p in printable)
        text = _clean(text)
        texts.append(text)

    return "raw", total, texts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assemble(pages: List[str]) -> str:
    """Join page texts with page-break markers.

    Args:
        pages: List of per-page text strings.

    Returns:
        Single assembled string.
    """
    parts: List[str] = []
    for i, text in enumerate(pages, start=1):
        if text.strip():
            parts.append(f"── Page {i} ──\n{text}")
    return "\n\n".join(parts)


def _clean(text: str) -> str:
    """Remove common PDF extraction artefacts from text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    # De-hyphenate across line breaks (word- \nword → word word)
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)
    # Collapse runs of spaces
    text = re.sub(r" {2,}", " ", text)
    # Normalise newlines
    text = re.sub(r"\r\n?", "\n", text)
    # Remove form-feed characters
    text = text.replace("\x0c", "\n")
    return text.strip()


def _slice_pages(pages: Any, start: int, end: Optional[int], total: int) -> Any:
    """Return a slice of pages using 1-based inclusive bounds.

    Args:
        pages: Sequence of page objects.
        start: First page number (1-based).
        end: Last page number (1-based) or None for all.
        total: Total page count.

    Returns:
        Sliced page sequence.
    """
    start_idx = max(0, start - 1)
    end_idx = min(total, end) if end is not None else total
    return pages[start_idx:end_idx]


def _assert_pdf(path: Path) -> None:
    """Check that a file begins with the PDF magic bytes.

    Args:
        path: File path.

    Raises:
        ValueError: If the file does not start with ``%PDF``.
    """
    header = path.read_bytes()[:4]
    if header != b"%PDF":
        raise ValueError(f"File is not a valid PDF (bad magic bytes): {path}")


def _resolve(path: str, project_root: str) -> Path:
    """Resolve a path to an absolute Path.

    Args:
        path: Raw path string.
        project_root: Optional project root.

    Returns:
        Resolved Path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if project_root:
        return (Path(project_root) / path).resolve()
    return (Path.cwd() / path).resolve()


def _relative(path: Path, project_root: str) -> str:
    """Compute a project-relative path string.

    Args:
        path: Absolute path.
        project_root: Project root directory.

    Returns:
        Relative path string or absolute path if not under root.
    """
    if not project_root:
        return str(path)
    try:
        return str(path.relative_to(Path(project_root)))
    except ValueError:
        return str(path)


def _try_import(name: str) -> Optional[Any]:
    """Attempt to import a module, returning None on failure.

    Args:
        name: Module name to import.

    Returns:
        Imported module or None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
