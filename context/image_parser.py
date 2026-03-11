"""image_parser.py — Sentinel @image: attachment handler.

Reads an image file from the local filesystem and returns a structured
attachment dict.  Images are always base64-encoded since they cannot be
represented as plain text.

For model consumption the attachment includes:
  - The base64-encoded image bytes
  - The detected MIME type
  - Pixel dimensions (width × height) when the standard `imghdr` /
    struct-based approach can determine them without a heavy dependency
  - A plain-text alt description field (empty by default — the LLM can
    populate it if needed)

No external dependencies — stdlib only (struct, imghdr where available).
"""

import base64
import mimetypes
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Maximum image size to base64-encode in full (50 MB).
_MAX_IMAGE_BYTES = 50 * 1024 * 1024

# Supported image MIME types
_SUPPORTED_MIMES = frozenset(
    {
        "image/png", "image/jpeg", "image/jpg", "image/gif",
        "image/webp", "image/bmp", "image/tiff", "image/svg+xml",
        "image/ico", "image/x-icon",
    }
)

# Extension → MIME fallbacks when mimetypes module returns None
_EXT_MIME: Dict[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff", ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(path: str, project_root: str = "") -> Dict[str, Any]:
    """Load an image file and return a structured attachment dict.

    Args:
        path: Absolute or project-relative path to the image file.
        project_root: Optional project root for relative path resolution.

    Returns:
        Attachment dict with keys:
            - ``type``        — always ``"image"``
            - ``path``        — resolved absolute file path
            - ``relative_path`` — path relative to project root
            - ``mime_type``   — MIME type string (e.g. ``"image/png"``)
            - ``content``     — base64-encoded image bytes (ASCII string)
            - ``encoding``    — always ``"base64"``
            - ``width``       — pixel width (int) or None if undetermined
            - ``height``      — pixel height (int) or None if undetermined
            - ``size_bytes``  — original file size in bytes
            - ``truncated``   — True if the file exceeded the size limit
            - ``alt``         — empty alt-text placeholder

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the file is not a recognised image type.
    """
    resolved = _resolve(path, project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")

    mime = _detect_mime(resolved)
    if mime not in _SUPPORTED_MIMES:
        raise ValueError(
            f"Unsupported image type '{mime}' for file: {resolved}\n"
            f"Supported types: {sorted(_SUPPORTED_MIMES)}"
        )

    size = resolved.stat().st_size
    raw = resolved.read_bytes()
    truncated = size > _MAX_IMAGE_BYTES
    raw = raw[:_MAX_IMAGE_BYTES]
    content = base64.b64encode(raw).decode("ascii")

    width, height = _read_dimensions(resolved, raw, mime)

    rel = _relative(resolved, project_root)

    return {
        "type": "image",
        "path": str(resolved),
        "relative_path": rel,
        "mime_type": mime,
        "content": content,
        "encoding": "base64",
        "width": width,
        "height": height,
        "size_bytes": size,
        "truncated": truncated,
        "alt": "",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(path: str, project_root: str) -> Path:
    """Resolve a potentially relative path to an absolute Path.

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


def _detect_mime(path: Path) -> str:
    """Detect the MIME type of an image file.

    Attempts mimetypes.guess_type first, then falls back to the extension
    mapping, then tries a byte-level magic-number check for common formats.

    Args:
        path: Image file path.

    Returns:
        MIME type string.
    """
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in _SUPPORTED_MIMES:
        return guessed

    ext_mime = _EXT_MIME.get(path.suffix.lower())
    if ext_mime:
        return ext_mime

    # Byte-level magic number fallback
    try:
        header = path.read_bytes()[:16]
        return _magic_mime(header) or "application/octet-stream"
    except OSError:
        return "application/octet-stream"


def _magic_mime(header: bytes) -> Optional[str]:
    """Identify image type from magic bytes in the file header.

    Args:
        header: First ≥ 16 bytes of the file.

    Returns:
        MIME string or None if unrecognised.
    """
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:2] in (b"\xff\xd8", b"\xff\xe0", b"\xff\xe1"):
        return "image/jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:2] in (b"BM",):
        return "image/bmp"
    if header[:4] in (b"MM\x00\x2a", b"II\x2a\x00"):
        return "image/tiff"
    return None


def _read_dimensions(
    path: Path, raw: bytes, mime: str
) -> Tuple[Optional[int], Optional[int]]:
    """Extract pixel dimensions from the image header bytes.

    Supports PNG, JPEG, GIF, BMP, and WebP without external libraries.
    Returns (None, None) for unsupported or unreadable formats.

    Args:
        path: File path (used for SVG text parsing).
        raw: Raw image bytes (possibly truncated).
        mime: MIME type string.

    Returns:
        Tuple of (width, height) in pixels, or (None, None).
    """
    try:
        if mime == "image/png":
            # PNG: IHDR chunk starts at byte 16, width=4 bytes, height=4 bytes
            if len(raw) >= 24:
                w = struct.unpack(">I", raw[16:20])[0]
                h = struct.unpack(">I", raw[20:24])[0]
                return w, h

        elif mime in ("image/jpeg", "image/jpg"):
            return _jpeg_dimensions(raw)

        elif mime == "image/gif":
            # GIF: bytes 6-9 are width/height as little-endian uint16
            if len(raw) >= 10:
                w = struct.unpack("<H", raw[6:8])[0]
                h = struct.unpack("<H", raw[8:10])[0]
                return w, h

        elif mime == "image/bmp":
            # BMP DIB header: width at offset 18, height at 22 (int32 LE)
            if len(raw) >= 26:
                w = struct.unpack("<i", raw[18:22])[0]
                h = abs(struct.unpack("<i", raw[22:26])[0])
                return w, h

        elif mime == "image/webp":
            return _webp_dimensions(raw)

        elif mime == "image/svg+xml":
            return _svg_dimensions(path)

    except (struct.error, ValueError, UnicodeDecodeError):
        pass

    return None, None


def _jpeg_dimensions(raw: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Parse JPEG SOF marker to extract dimensions.

    Args:
        raw: JPEG file bytes.

    Returns:
        (width, height) or (None, None).
    """
    i = 2  # skip initial FF D8
    while i < len(raw) - 9:
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        # SOF markers: C0–C3, C5–C7, C9–CB, CD–CF
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = struct.unpack(">H", raw[i + 5: i + 7])[0]
            w = struct.unpack(">H", raw[i + 7: i + 9])[0]
            return w, h
        length = struct.unpack(">H", raw[i + 2: i + 4])[0]
        i += 2 + length
    return None, None


def _webp_dimensions(raw: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Parse WebP header to extract dimensions.

    Args:
        raw: WebP file bytes.

    Returns:
        (width, height) or (None, None).
    """
    if len(raw) < 30:
        return None, None
    # Lossy VP8: chunk header at 12, width at 26-27 (14 bits), height at 28-29
    if raw[12:16] == b"VP8 ":
        w = (struct.unpack("<H", raw[26:28])[0] & 0x3FFF) + 1
        h = (struct.unpack("<H", raw[28:30])[0] & 0x3FFF) + 1
        return w, h
    # Lossless VP8L: signature byte 0x2F then width-1 (14 bits) and height-1
    if raw[12:16] == b"VP8L" and len(raw) >= 25 and raw[20] == 0x2F:
        bits = struct.unpack("<I", raw[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    return None, None


def _svg_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Parse SVG XML for width/height attributes.

    Args:
        path: Path to the SVG file.

    Returns:
        (width, height) as integers, or (None, None).
    """
    import re
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        w_match = re.search(r'<svg[^>]+width=["\'](\d+)', text)
        h_match = re.search(r'<svg[^>]+height=["\'](\d+)', text)
        if w_match and h_match:
            return int(w_match.group(1)), int(h_match.group(1))
    except OSError:
        pass
    return None, None


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
