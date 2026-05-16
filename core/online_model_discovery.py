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
# Domains that map to the "coding" agent (for parameter conditioning)
# ---------------------------------------------------------------------------

_CODING_AGENT_DOMAINS: frozenset = frozenset({
    "coding_frontend",
    "coding_backend",
    "coding_general",
    "data_science",
    "creative",
    "other",  # fallback for unknown domains typically goes to coding
    "root",   # project root initialization typically goes to coding
})


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


def _extract_param_count_from_model(model: Dict[str, Any], tag: str) -> float:
    """Extract parameter count in billions from a model dict.

    Tries multiple sources: details metadata, model tag string, and fallback.

    Args:
        model: Model dict from cloud API response.
        tag:   Model tag/name string.

    Returns:
        Parameter count in billions, or 0.0 if cannot be determined.
    """
    # Try structured details field first
    details = model.get("details", {}) or {}
    size_str = details.get("parameter_size") or details.get("parameters") or ""
    if size_str:
        return _parse_param_billions(str(size_str))

    # Try extracting from tag string
    clean = tag.replace(":cloud", "").replace("-cloud", "").upper()
    # MoE pattern: 8x7b
    moe = re.search(r"(\d+)X(\d+(?:\.\d+)?)B", clean)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    # Plain pattern: 31b, 7b, 70b
    plain = re.search(r"(\d+(?:\.\d+)?)B", clean)
    if plain:
        return float(plain.group(1))

    return 0.0


def _should_exclude_model_for_coding(model_tag: str, model: Dict[str, Any]) -> bool:
    """Check if a model should be excluded from coding tasks due to size.

    Models with > 50B parameters are reserved for reasoning tasks.
    Coding tasks benefit from smaller, faster models.

    Args:
        model_tag: The model identifier string.
        model:     The model dict from cloud API.

    Returns:
        True if the model should be excluded for coding tasks, False otherwise.
    """
    param_count = _extract_param_count_from_model(model, model_tag)
    # Exclude models with more than 50B parameters for coding
    return param_count > 50.0


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

    def _is_coding_task(self, domain: str) -> bool:
        """Check if the domain maps to a coding agent task.

        Args:
            domain: The task's routing_domain or domain value.

        Returns:
            True if the task will be assigned to a coding agent, False otherwise.
        """
        return domain in _CODING_AGENT_DOMAINS

    def _filter_models_for_task(self, available_tags: List[str], domain: str) -> List[str]:
        """Filter available models based on task type and model constraints.

        In online mode, very large models (>50B parameters) should not be used
        for coding tasks, as they are expensive and slower. Instead, they are
        reserved for reasoning tasks where their extra capacity is beneficial.

        Args:
            available_tags: List of available model tags.
            domain:         The task's routing_domain.

        Returns:
            Filtered list of model tags, excluding those that violate constraints.
        """
        # If this is a coding task, exclude models with >50B parameters
        if not self._is_coding_task(domain):
            return available_tags

        # Build tag-to-model lookup
        models_by_tag = {m.get("name") or m.get("id") or m.get("model", ""): m
                         for m in self._get_cloud_models()}

        filtered = []
        for tag in available_tags:
            model = models_by_tag.get(tag, {})
            if not _should_exclude_model_for_coding(tag, model):
                filtered.append(tag)

        # If all models were filtered out (edge case), return the original list
        # to allow fallback chain to work
        return filtered if filtered else available_tags

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

        In online mode for coding tasks, models with >50B parameters are excluded
        and reserved for reasoning agents.

        No model names or scores are hardcoded here; everything is inferred
        from what the Ollama Cloud API returns and what current benchmarks say.
        """
        if self._cloud is None:
            return None

        available_models = self._get_cloud_models()
        available_tags   = self._cloud_model_tags()
        if not available_tags:
            return None

        # ── Apply task-specific model filtering ─────────────────────────────
        # For coding tasks, exclude very large models (>50B parameters)
        available_tags = self._filter_models_for_task(available_tags, domain)
        tag_to_meta: Dict[str, Dict] = {}
        for m in available_models:
            tag = m.get("name") or m.get("id") or m.get("model", "")
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

        # Note in reason if large models were filtered for coding tasks
        if self._is_coding_task(domain):
            reason += " [large models excluded for coding]"

        return best_base, reason

    # ------------------------------------------------------------------
    # Decomposition model selector (≤250B cap)
    # ------------------------------------------------------------------

    def select_decomposition_model(
        self,
        domain: str,
        complexity: str,
    ) -> Optional[Tuple[str, str]]:
        """Select the best cloud model for task decomposition.

        Uses the same two-layer scoring as _select_ollama_cloud() but:
        1. Treats the task as a reasoning task regardless of domain
           (decomposition is always a structured reasoning operation).
        2. Hard-filters any model whose parameter count exceeds 250B.
           If all models exceed 250B, fall back to the full unfiltered list
           so we always have a result.
        3. Adjusts _complexity_size_score to use "medium" cap for the size
           dimension (we don't want the very largest models, just capable ones).

        Returns (model_base_tag, reason) or None if no cloud models available.
        """
        available_models = self._get_cloud_models()
        if not available_models:
            return None

        # Build tag list from available models
        all_candidates = []
        for m in available_models:
            tag = m.get("name") or m.get("id") or m.get("model", "")
            if tag:
                all_candidates.append((tag, m))

        if not all_candidates:
            return None

        # Filter to models ≤250B; fall back to full list if all exceed cap
        decomp_candidates = [
            (tag, m) for tag, m in all_candidates
            if _extract_param_count_from_model(m, tag) <= 250.0
        ]
        if not decomp_candidates:
            decomp_candidates = all_candidates

        # Layer 1: metadata scoring — always treat as "reasoning" for decomposition
        metadata_scores: Dict[str, float] = {}
        for tag, meta in decomp_candidates:
            details = meta.get("details", {}) or {}

            param_b = _extract_param_count_from_model(meta, tag)

            # Family keywords from API metadata
            family = (
                details.get("family", "")
                + " "
                + details.get("format", "")
            ).lower()

            score = 0.0
            # Use "medium" cap for size dimension — capable but not extreme
            score += _complexity_size_score(param_b, "medium")
            # Reasoning keyword score (decomposition is always reasoning)
            score += _name_keyword_score(tag, "reasoning")
            # Family keyword bonus for reasoning domain
            for kw in _DOMAIN_KEYWORDS.get("reasoning", []):
                if kw in family:
                    score += 0.5

            metadata_scores[tag] = score

        # Layer 2: web-search scoring — use actual subtask complexity for relevance
        web_scores = self._web_search_scores("reasoning", complexity)

        # Combine scores
        combined: Dict[str, float] = {}
        for tag, _ in decomp_candidates:
            base_tag = tag.replace(":cloud", "").replace("-cloud", "")
            combined[tag] = (
                metadata_scores.get(tag, 0.0)
                + web_scores.get(base_tag, 0.0)
            )

        if not combined:
            return None

        best_tag = max(combined, key=combined.__getitem__)
        best_score = combined[best_tag]
        best_base = best_tag.replace(":cloud", "").replace("-cloud", "")

        meta_s = metadata_scores.get(best_tag, 0.0)
        search_s = web_scores.get(best_base, 0.0)
        best_meta = dict(decomp_candidates)[best_tag] if best_tag in dict(decomp_candidates) else {}
        param_b_best = _extract_param_count_from_model(best_meta, best_tag)

        reason = (
            f"decomposition model for {domain}/{complexity} "
            f"(params={param_b_best:.0f}B meta={meta_s:.2f} "
            f"web={search_s:.2f} total={best_score:.2f} ≤250B cap applied)"
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

        Model conditioning for coding tasks
        ------------------------------------
        In online mode, very large models (>50B parameters) are excluded from
        coding tasks and reserved for reasoning agents. This is because:
        - Large models are slower and more expensive for routine coding tasks
        - Smaller, specialized models are often more efficient for code generation
        - Reasoning tasks benefit from the extra capacity for complex analysis

        The filtering is applied during cloud model selection; if no suitable
        models remain after filtering, the original list is used as fallback.

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


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------


def pick_decomposition_model(
    cloud_models: List[Dict],
    domain: str = "reasoning",
    complexity: str = "medium",
    tool_registry: Optional[Any] = None,
    ttl_seconds: int = 1800,
) -> str:
    """Convenience wrapper: builds a temporary engine and calls
    select_decomposition_model(). Returns the model tag string, or ''.

    Args:
        cloud_models:  List of model dicts returned by OllamaCloudClient.list_models().
        domain:        Domain hint for web-search scoring (default: "reasoning").
        complexity:    Complexity level for scoring (default: "medium").
        tool_registry: Optional tool registry for web-search scoring.
        ttl_seconds:   Cache TTL in seconds (default: 1800).

    Returns:
        The model base tag string, or '' if no model could be selected.
    """
    engine = OnlineModelDiscoveryEngine(
        ollama_cloud_client=None,
        tool_registry=tool_registry,
        ttl_seconds=ttl_seconds,
    )
    # Manually populate the cache so _get_cloud_models() returns our list
    engine._cached_models = cloud_models
    engine._cache_ts = time.monotonic()
    result = engine.select_decomposition_model(domain, complexity)
    return result[0] if result else ""
