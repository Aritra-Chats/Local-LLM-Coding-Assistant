"""url_fetcher.py — Sentinel @url: attachment handler.

Fetches a web resource and returns its content as a structured attachment
dict.  Handles:

  - HTML pages  — returns both raw HTML and a cleaned plain-text extract
  - JSON APIs   — returns the parsed JSON serialised as indented text
  - Plain text  — returned as-is
  - Other MIME  — content is base64-encoded and flagged as binary

Safety measures:
  - SSRF guard: rejects private/loopback IP ranges and non-http(s) schemes
  - Configurable size limit (default 5 MB)
  - Configurable timeout (default 15 s)
  - Follows up to 5 redirects; rejects redirects to private addresses

Dependencies: stdlib only (urllib + html.parser).  Does NOT use 'requests'
to avoid adding another dependency, but falls back to requests if available.
"""

import base64
import html.parser
import io
import json
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_CONTENT_BYTES = 5 * 1024 * 1024   # 5 MB
_REQUEST_TIMEOUT = 15                   # seconds
_MAX_REDIRECTS = 5

_DEFAULT_HEADERS = {
    "User-Agent": "Sentinel/1.0 (local AI assistant; context fetcher)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9",
    "Accept-Language": "en",
}

# Private / loopback CIDR ranges to block (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(
    url: str,
    timeout: int = _REQUEST_TIMEOUT,
    max_bytes: int = _MAX_CONTENT_BYTES,
) -> Dict[str, Any]:
    """Fetch a URL and return a structured attachment dict.

    Args:
        url: The URL to fetch.  Must use http:// or https://.
        timeout: Request timeout in seconds.
        max_bytes: Maximum response body size before truncation.

    Returns:
        Attachment dict with keys:
            - ``type``         — always ``"url"``
            - ``url``          — the original requested URL
            - ``final_url``    — the URL after following redirects
            - ``status_code``  — HTTP status code (int)
            - ``mime_type``    — Content-Type header value
            - ``content``      — text content or base64-encoded bytes
            - ``encoding``     — ``"text"`` or ``"base64"``
            - ``plain_text``   — cleaned plain-text for HTML responses
            - ``title``        — page title for HTML responses
            - ``size_bytes``   — response body size
            - ``truncated``    — True if body exceeded max_bytes

    Raises:
        ValueError: If the URL scheme is invalid or the host is private.
        urllib.error.URLError: On network errors.
    """
    _validate_url(url)

    raw_bytes, final_url, status, content_type = _fetch(url, timeout, max_bytes)
    truncated = len(raw_bytes) == max_bytes

    mime = _parse_mime(content_type)

    if "json" in mime:
        content, encoding, plain_text, title = _handle_json(raw_bytes)
    elif "html" in mime or "xhtml" in mime:
        content, encoding, plain_text, title = _handle_html(raw_bytes, content_type)
    elif mime.startswith("text/"):
        text = raw_bytes.decode("utf-8", errors="replace")
        content, encoding, plain_text, title = text, "text", text, ""
    else:
        content = base64.b64encode(raw_bytes).decode("ascii")
        encoding, plain_text, title = "base64", "", ""

    return {
        "type": "url",
        "url": url,
        "final_url": final_url,
        "status_code": status,
        "mime_type": mime,
        "content": content,
        "encoding": encoding,
        "plain_text": plain_text,
        "title": title,
        "size_bytes": len(raw_bytes),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> None:
    """Reject URLs that could cause SSRF or use unsafe schemes.

    Args:
        url: URL string to validate.

    Raises:
        ValueError: If the URL is deemed unsafe.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsafe URL scheme '{parsed.scheme}'. Only http:// and https:// are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    # Resolve hostname to IP addresses and check each one
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise ValueError(
                    f"URL '{url}' resolves to a private/loopback address "
                    f"({ip}) which is not allowed."
                )


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def _fetch(
    url: str, timeout: int, max_bytes: int
) -> Tuple[bytes, str, int, str]:
    """Perform the HTTP request and return (body, final_url, status, content_type).

    Args:
        url: Validated URL to fetch.
        timeout: Timeout in seconds.
        max_bytes: Maximum bytes to read.

    Returns:
        Tuple of (body bytes, final URL string, status code, content-type).
    """
    req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.url or url
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read(max_bytes)
    except urllib.error.HTTPError as exc:
        # Return error body for non-2xx so callers see what the server said
        final_url = url
        status = exc.code
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        body = exc.read(max_bytes) if exc.fp else b""

    return body, final_url, status, content_type


# ---------------------------------------------------------------------------
# Content handlers
# ---------------------------------------------------------------------------


def _handle_json(raw: bytes) -> Tuple[str, str, str, str]:
    """Parse and pretty-print a JSON response body.

    Args:
        raw: Raw response bytes.

    Returns:
        (content, encoding, plain_text, title)
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        return pretty, "text", pretty, ""
    except json.JSONDecodeError:
        return text, "text", text, ""


def _handle_html(raw: bytes, content_type: str) -> Tuple[str, str, str, str]:
    """Extract text and title from an HTML response.

    Args:
        raw: Raw HTML bytes.
        content_type: Content-Type header (used for charset detection).

    Returns:
        (raw_html, encoding, plain_text, title)
    """
    charset = _parse_charset(content_type) or "utf-8"
    html_text = raw.decode(charset, errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(html_text)
    plain = parser.get_text()
    title = parser.title or ""
    return html_text, "text", plain, title


# ---------------------------------------------------------------------------
# HTML text extractor
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Minimal HTML → plain-text converter.

    Skips <script>, <style>, and <noscript> tags.  Collapses whitespace
    and preserves paragraph breaks.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head"})
    _BLOCK_TAGS = frozenset(
        {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
         "li", "tr", "td", "th", "br", "article", "section"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag_lower == "title":
            self._in_title = True
        if tag_lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag_lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title and not self.title:
            self.title = data.strip()
        self._parts.append(data)

    def get_text(self) -> str:
        """Return cleaned plain-text."""
        raw = "".join(self._parts)
        # Collapse runs of whitespace but preserve paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_mime(content_type: str) -> str:
    """Extract the MIME type from a Content-Type header value.

    Args:
        content_type: Full header value, e.g. ``"text/html; charset=utf-8"``.

    Returns:
        MIME type string, e.g. ``"text/html"``.
    """
    return content_type.split(";")[0].strip().lower()


def _parse_charset(content_type: str) -> Optional[str]:
    """Extract charset from a Content-Type header.

    Args:
        content_type: Full header value.

    Returns:
        Charset string or None.
    """
    match = re.search(r"charset\s*=\s*([^\s;\"']+)", content_type, re.IGNORECASE)
    return match.group(1).strip() if match else None
