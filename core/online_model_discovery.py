"""online_model_discovery.py — Per-sub-task model selection for online mode.

Selection priority (per sub-task):

  1. **Ollama Cloud**  — fetches available cloud models via ``GET /api/tags``
     (cached 30 min), scores every available model for the task's domain and
     complexity using a two-layer approach:

       a. *Metadata scoring* — parameter size, model family, and name keywords
          extracted from the live API response.  No values are hardcoded; all
          signals come from what the API returns at runtime.

       b. *Web-search scoring* — a single search per (domain, complexity) pair
          asks for benchmark comparisons; every available model is scored by
          how frequently and how highly it appears across all result snippets.
          Earlier results are weighted more heavily.

     The model with the highest combined score is selected.

  2. **External providers** — Anthropic → Google → OpenAI (affinity-guided).
  3. **Offline degradation** — falls back to :class:`ConcreteModelRouter`.

Cloud model tags used with the direct API do NOT carry the ':cloud' suffix
(that is only for local-daemon usage).
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Domain → preferred external provider (affinity, used only as fallback)
# ---------------------------------------------------------------------------

_DOMAIN_AFFINITY: Dict[str, str] = {
    "math":            "google",
    "data_science":    "google",
    "research":        "anthropic",
    "reasoning":       "anthropic",
    "security":        "anthropic",
    "creative":        "anthropic",
    "coding_frontend": "openai",
    "coding_backend":  "openai",
    "coding_general":  "openai",
    "debugging":       "openai",
    "devops":          "openai",
}

# ---------------------------------------------------------------------------
# Domain → keyword signals used to score model names / descriptions
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding_frontend":  ["frontend", "ui", "web", "react", "vue", "css", "html",
                         "javascript", "typescript", "design", "visual"],
    "coding_backend":   ["backend", "server", "api", "database", "sql", "rest",
                         "graphql", "microservice"],
    "coding_general":   ["code", "coding", "programming", "developer", "software"],
    "debugging":        ["debug", "analysis", "reasoning", "logic", "thinking"],
    "devops":           ["devops", "infra", "docker", "kubernetes", "ci", "cloud",
                         "deploy"],
    "data_science":     ["data", "science", "analysis", "math", "statistics",
                         "ml", "machine learning"],
    "security":         ["security", "safety", "audit", "vulnerability"],
    "research":         ["research", "reasoning", "analysis", "science"],
    "math":             ["math", "reasoning", "logic", "science"],
    "creative":         ["creative", "writing", "story", "content"],
    "system":           ["system", "os", "kernel", "infra"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_param_billions(size_str: str) -> float:
    """Parse a parameter-size string like '7B', '671b', '8x7B' → float billions.

    Returns 0.0 if the string cannot be parsed.
    """
    if not size_str:
        return 0.0
    s = size_str.strip().upper()
    # Mixture-of-experts notation: 8x7B → 56 B effective
    moe = re.match(r"(\d+)X(\d+(?:\.\d+)?)B", s)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    # Standard notation: 7B, 671b, 1.5B, 120B
    plain = re.match(r"(\d+(?:\.\d+)?)B", s)
    if plain:
        return float(plain.group(1))
    return 0.0


def _name_keyword_score(model_tag: str, domain: str) -> float:
    """Return a score [0, 3] based on how many domain keywords appear in the model name."""
    base = model_tag.replace(":cloud", "").lower()
    keywords = _DOMAIN_KEYWORDS.get(domain, [])
    hits = sum(1 for kw in keywords if kw in base)
    return min(hits * 1.0, 3.0)


def _complexity_size_score(param_b: float, complexity: str) -> float:
    """Score a model by how well its size matches the required complexity.

    * high complexity  → reward larger models (more parameters = more capable)
    * low complexity   → slightly reward smaller / faster models
    * medium           → moderate reward scaling with size
    """
    if param_b <= 0.0:
        return 0.0
    if complexity == "high":
        # Sigmoid-like: caps at ~3 for 70 B+
        return min(param_b / 70.0 * 3.0, 3.0)
    elif complexity == "low":
        # Prefer lighter models — penalise very large ones slightly
        return max(0.0, 1.5 - param_b / 100.0)
    else:
        # Medium: gentle linear reward
        return min(param_b / 100.0 * 2.0, 2.0)


class OnlineModelDiscoveryEngine:
    """Selects the best cloud model for each sub-task.

    Parameters
    ----------
    ollama_cloud_client:
        :class:`~models.ollama_cloud_client.OllamaCloudClient` instance.
        May be ``None`` if ``OLLAMA_API_KEY`` is not set.
    local_router:
        :class:`~core.model_router.ConcreteModelRouter` for offline fallback.
    tool_registry:
        Optional tool registry; if a ``web_search`` tool is present it is
        used to fetch benchmark recommendations.
    ttl_seconds:
        Cloud model list cache TTL (default 1800 = 30 minutes).
    """

    def __init__(
        self,
        ollama_cloud_client: Optional[Any] = None,
        local_router: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        ttl_seconds: int = 1800,
    ) -> None:
        self._cloud   = ollama_cloud_client
        self._router  = local_router
        self._tools   = tool_registry
        self._ttl     = ttl_seconds

        self._cached_models: Optional[List[Dict]] = None
        self._cache_ts: float = 0.0
        # Cache web-search scores per (domain, complexity) to avoid duplicate queries
        self._search_score_cache: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Cached cloud model list
    # ------------------------------------------------------------------

    def _get_cloud_models(self) -> List[Dict]:
        now = time.monotonic()
        if self._cached_models is not None and (now - self._cache_ts) < self._ttl:
            return self._cached_models
        if self._cloud is None:
            return []
        try:
            models = self._cloud.list_models()
            self._cached_models = models
            self._cache_ts = now
            return models
        except Exception:
            return []

    def _cloud_model_tags(self) -> List[str]:
        # Support both /api/tags ("name") and /v1/models OpenAI format ("id")
        tags = []
        for m in self._get_cloud_models():
            tag = m.get("name") or m.get("id") or m.get("model", "")
            if tag:
                tags.append(tag)
        return tags

    # ------------------------------------------------------------------
    # Web-search scoring
    # ------------------------------------------------------------------

    def _web_search_scores(self, domain: str, complexity: str) -> Dict[str, float]:
        """Return a score dict {model_base_tag: score} derived from web search.

        All available models are scored by how often and how prominently they
        appear in benchmark comparison snippets for this domain + complexity.
        Results are cached per (domain, complexity) pair within the session.

        The scoring formula:
          - Each snippet mentioning a model contributes
            (n_results - result_rank) / n_results * 2.0 points
            so earlier search results carry more weight.
          - Multiple snippets accumulate; there is no per-snippet cap.
        """
        cache_key = f"{domain}:{complexity}"
        if cache_key in self._search_score_cache:
            return self._search_score_cache[cache_key]

        scores: Dict[str, float] = {}
        available_tags = self._cloud_model_tags()
        if not available_tags or self._tools is None:
            return scores

        try:
            query = (
                f"best large language model {domain} tasks benchmark comparison "
                f"{complexity} complexity 2025"
            )
            ws_result = self._tools.invoke("web_search", {"query": query, "max_results": 8})
            output = ws_result.get("output", [])
            if not isinstance(output, list):
                return scores

            n = max(len(output), 1)
            for rank, item in enumerate(output):
                snippet = (
                    str(item.get("snippet", ""))
                    + " "
                    + str(item.get("title", ""))
                ).lower()
                weight = (n - rank) / n * 2.0  # 2.0 → 0.25 across 8 results

                for tag in available_tags:
                    base = tag.replace(":cloud", "")
                    if base.lower() in snippet:
                        scores[base] = scores.get(base, 0.0) + weight
        except Exception:
            pass

        self._search_score_cache[cache_key] = scores
        return scores

    # ------------------------------------------------------------------
    # Main Ollama Cloud selection
    # ------------------------------------------------------------------

    def _select_ollama_cloud(
        self,
        domain: str,
        complexity: str,
    ) -> Optional[Tuple[str, str]]:
        """Return (model_base_tag, reason) for the highest-scoring cloud model.

        Scoring layers (all derived at runtime from live API data + web search):

        1. ``metadata_score`` — parameter size × complexity fit + name keywords
        2. ``search_score``   — web benchmark mention frequency + position rank

        No model names or scores are hardcoded here; everything is inferred
        from what the Ollama Cloud API returns and what current benchmarks say.
        """
        if self._cloud is None:
            return None

        available_models = self._get_cloud_models()
        available_tags   = self._cloud_model_tags()
        if not available_tags:
            return None

        # ── Layer 1: Metadata scoring ──────────────────────────────────────
        # Build a lookup: tag → model dict for parameter-size extraction
        tag_to_meta: Dict[str, Dict] = {}
        for m in available_models:
            tag = m.get("name", m.get("model", ""))
            if tag:
                tag_to_meta[tag] = m

        metadata_scores: Dict[str, float] = {}
        for tag in available_tags:
            meta   = tag_to_meta.get(tag, {})
            details = meta.get("details", {})

            # Extract parameter count from model metadata
            param_size_str = (
                details.get("parameter_size")
                or details.get("parameters")
                or ""
            )
            # Extract size from the full tag when not in details metadata.
            # Use re.search over the whole string so tags like "gemma4:31b-cloud"
            # and "qwen3.5:32b" are handled — the old split(":")[-1] approach
            # would return "cloud" for tags ending in ":cloud" giving 0.0.
            if not param_size_str:
                clean = re.sub(r"[:-]cloud$", "", tag, flags=re.IGNORECASE)
                moe_m = re.search(r"(\d+)[xX](\d+(?:\.\d+)?)b", clean, re.IGNORECASE)
                if moe_m:
                    param_size_str = f"{float(moe_m.group(1)) * float(moe_m.group(2))}b"
                else:
                    plain_m = re.search(r"(\d+(?:\.\d+)?)b", clean, re.IGNORECASE)
                    if plain_m:
                        param_size_str = plain_m.group(0)

            param_b = _parse_param_billions(str(param_size_str))

            # Model family from API metadata (e.g. "llama", "mistral")
            family = (
                details.get("family", "")
                + " "
                + details.get("format", "")
            ).lower()

            score = 0.0
            # a) Complexity ↔ size fit
            score += _complexity_size_score(param_b, complexity)
            # b) Domain keywords in model name
            score += _name_keyword_score(tag, domain)
            # c) Domain keywords in model family metadata
            for kw in _DOMAIN_KEYWORDS.get(domain, []):
                if kw in family:
                    score += 0.5

            metadata_scores[tag] = score

        # ── Layer 2: Web-search scoring ────────────────────────────────────
        search_scores = self._web_search_scores(domain, complexity)

        # ── Combine scores ─────────────────────────────────────────────────
        combined: Dict[str, float] = {}
        for tag in available_tags:
            base = tag.replace(":cloud", "")
            combined[tag] = (
                metadata_scores.get(tag, 0.0)
                + search_scores.get(base, 0.0)
            )

        if not combined:
            return None

        best_tag   = max(combined, key=combined.__getitem__)
        best_score = combined[best_tag]
        best_base  = best_tag.replace(":cloud", "")

        # Build a short explanation for the table display
        meta_s   = metadata_scores.get(best_tag, 0.0)
        search_s = search_scores.get(best_base, 0.0)
        reason = (
            f"runtime-scored for {domain}/{complexity} "
            f"(meta={meta_s:.2f} web={search_s:.2f} total={best_score:.2f})"
        )
        return best_base, reason

    # ------------------------------------------------------------------
    # External provider fallback
    # ------------------------------------------------------------------

    def _select_external(self, domain: str) -> Optional[Tuple[str, str, str]]:
        """Return (provider, model, reason) for the best available external provider."""
        import os

        affinity = _DOMAIN_AFFINITY.get(domain, "")
        order    = [affinity] if affinity else []
        for fallback in ("anthropic", "google", "openai"):
            if fallback not in order:
                order.append(fallback)

        from models.external_api_client import ExternalAPIClient
        for provider in order:
            client = ExternalAPIClient(provider)
            if client.is_available():
                meta  = ExternalAPIClient.PROVIDERS[provider]
                model = os.environ.get(meta["model_env"], "") or meta["default"]
                return provider, model, "external fallback (ollama cloud unavailable)"

        return None

    # ------------------------------------------------------------------
    # Main discovery method
    # ------------------------------------------------------------------

    def discover(self, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best model for *sub_task* and store it in-place.

        Populates ``sub_task["selected_model"]`` with::

            {
                "provider": "ollama_cloud"|"anthropic"|"openai"|"google"|"ollama_local",
                "model":    "<tag or model id>",
                "reason":   "<justification string>",
            }

        Args:
            sub_task: A sub-task dict (should already have ``routing_domain``).

        Returns:
            The mutated *sub_task* dict.
        """
        domain     = sub_task.get("routing_domain", sub_task.get("domain", "other"))
        complexity = sub_task.get("complexity", "medium")

        # ── Priority 1: Ollama Cloud (scored) ─────────────────────────
        cloud_result = self._select_ollama_cloud(domain, complexity)
        if cloud_result:
            tag, reason = cloud_result
            sub_task["selected_model"] = {
                "provider": "ollama_cloud",
                "model":    tag,
                "reason":   reason,
            }
            return sub_task

        # ── Priority 2: External provider fallback ─────────────────────
        ext_result = self._select_external(domain)
        if ext_result:
            provider, model, reason = ext_result
            sub_task["selected_model"] = {
                "provider": provider,
                "model":    model,
                "reason":   reason,
            }
            return sub_task

        # ── Priority 3: Offline degradation ───────────────────────────
        try:
            if self._router:
                local_model = self._router.select_coding_model()
                sub_task["selected_model"] = {
                    "provider": "ollama_local",
                    "model":    local_model,
                    "reason":   "no cloud/external provider available — offline degradation",
                }
                return sub_task
        except Exception:
            pass

        sub_task["selected_model"] = {
            "provider": "ollama_local",
            "model":    "",
            "reason":   "degraded — no model could be selected",
        }
        return sub_task
