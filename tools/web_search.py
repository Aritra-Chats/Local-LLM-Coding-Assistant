"""web_search.py — Sentinel web_search tool.

Performs a web search using DuckDuckGo's public HTML interface (no API key
required) and returns the top result snippets.  Falls back gracefully to
returning the search URL when the HTTP request fails.

Security note
-------------
All outbound requests are made to ``https://html.duckduckgo.com`` only.
User-supplied queries are URL-encoded; no SSRF surface is introduced.

Registered name: ``"web_search"``
"""

import html
import re
import urllib.parse
import urllib.request
from typing import Any, List

from tools.tool_registry import Tool, ToolResult

_DDG_URL = "https://html.duckduckgo.com/html/"
_DEFAULT_MAX_RESULTS = 5
_REQUEST_TIMEOUT = 15  # seconds

# Simple pattern to strip HTML tags
_TAG_RE = re.compile(r"<[^>]+>")
# Result blocks on DDG HTML: title + snippet + URL
_RESULT_RE = re.compile(
    r'class="result__title"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(.*?)</(?:a|div|span)>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _ddg_search(query: str, max_results: int) -> List[dict]:
    """Fetch DuckDuckGo HTML results and return a list of result dicts."""
    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    req = urllib.request.Request(
        _DDG_URL,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Sentinel/1.0)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310
        body = resp.read(512 * 1024).decode("utf-8", errors="replace")

    results = []
    for m in _RESULT_RE.finditer(body):
        url = _strip_tags(m.group(1))
        title = _strip_tags(m.group(2))
        snippet = _strip_tags(m.group(3))
        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo and return result snippets.

    Parameters:
        query (str, required): The search query.
        max_results (int, optional): Maximum number of results to return.
            Defaults to 5.
    """

    name = "web_search"
    description = (
        "Search the web via DuckDuckGo and return the top result titles, "
        "URLs, and snippets.  No API key required."
    )
    parameters_schema = {
        "query": {
            "type": "string",
            "description": "The search query.",
            "required": True,
        },
        "max_results": {
            "type": "int",
            "description": f"Maximum number of results to return.  Defaults to {_DEFAULT_MAX_RESULTS}.",
            "required": False,
            "default": _DEFAULT_MAX_RESULTS,
        },
    }

    def run(  # type: ignore[override]
        self,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
        **_: Any,
    ) -> ToolResult:
        """Execute the web search.

        Args:
            query: Search query string.
            max_results: Number of results to return.

        Returns:
            ToolResult with ``output`` as a list of
            ``{"title": str, "url": str, "snippet": str}`` dicts.
            On network failure, ``output`` contains the fallback search URL.
        """
        max_results = max(1, min(int(max_results), 20))
        search_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"

        try:
            results = _ddg_search(query, max_results)
        except Exception as exc:  # noqa: BLE001
            # Return the plain search URL as a fallback — caller can open it.
            return ToolResult(
                tool_name=self.name,
                success=False,
                output={"fallback_url": search_url},
                error=f"Search request failed: {exc}",
                metadata={"query": query},
            )

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=[],
                metadata={"query": query, "fallback_url": search_url},
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=results,
            metadata={"query": query, "total_returned": len(results)},
        )
