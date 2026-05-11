#!/usr/bin/env python3
"""cloud_model_probe.py — Discover accessible Ollama Cloud models.

Probes Ollama Cloud to determine which models are actually accessible under
the current API key's subscription plan, then writes results to
~/.sentinel/cloud_models.json.

Usage
-----
    python scripts/cloud_model_probe.py            # respects 24-hour TTL
    python scripts/cloud_model_probe.py --force    # always regenerates

Called automatically by:
  - Windows Task Scheduler / cron  (daily, at 06:00)
  - ``sentinel refresh``           (on-demand, always forces regeneration)

How it works
------------
1.  Reads OLLAMA_API_KEY from ~/.sentinel/.env or the environment.
    If the key is absent the script exits cleanly — it will try again on
    the next scheduled run once the key has been configured.
2.  Checks connectivity to https://ollama.com.  If the host is unreachable
    the cache is left intact and the script exits cleanly.
3.  Lists all models from the Ollama Cloud API using the ``ollama`` SDK.
4.  Sends a 1-token generate probe to each candidate:
      - HTTP 200           → accessible on this plan   (✓)
      - HTTP 401/402/403   → blocked by subscription   (✗)
      - Other error        → uncertain (included with a warning flag)
5.  Writes ~/.sentinel/cloud_models.json, sorted by parameter count so
    callers always get the most capable model first.

Cache format (cloud_models.json)
---------------------------------
    {
      "version":      2,
      "generated_at": "<ISO-8601>",
      "expires_at":   "<ISO-8601>",        # generated_at + 24 h
      "api_key_hash": "<sha256 prefix>",   # invalidated when key changes
      "models": [
        {
          "name":           "gemma4:31b",      # bare API tag (no :cloud suffix)
          "cloud_tag":      "gemma4:31b-cloud",# local-daemon tag
          "param_billions": 31.0,
          "probe_status":   "ok"               # "ok" | "uncertain"
        }
      ],
      "inaccessible": ["mistral-large-3:675b", ...]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTINEL_HOME  = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
CACHE_FILE     = SENTINEL_HOME / "cloud_models.json"
CACHE_TTL_HRS  = 24
CLOUD_HOST     = "https://ollama.com"

# Minimal generate payload used for probing — 1 token, deterministic
_PROBE_OPTS = {"num_predict": 1, "temperature": 0.0}


# ---------------------------------------------------------------------------
# Cloud suffix helper
# ---------------------------------------------------------------------------
def to_cloud_tag(api_tag: str) -> str:
    """Convert a bare API tag to its local-daemon cloud variant.

    Rule:
        ``gemma4:31b``   → ``gemma4:31b-cloud``   (has size variant → ``-cloud``)
        ``kimi-k2.6``    → ``kimi-k2.6:cloud``    (no variant       → ``:cloud``)

    The cloud suffix is used only when running a model via the local Ollama
    daemon (``ollama run gemma4:31b-cloud``).  The direct Ollama Cloud API
    (https://ollama.com/api/generate) always uses the bare tag.
    """
    return f"{api_tag}-cloud" if ":" in api_tag else f"{api_tag}:cloud"


# ---------------------------------------------------------------------------
# .env reader  (mirrors core/api_key_manager.py, no project import needed)
# ---------------------------------------------------------------------------
def _load_env_file(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key
    return _load_env_file(SENTINEL_HOME / ".env").get("OLLAMA_API_KEY") or None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _key_hash(key: str) -> str:
    """Return a short hash of the API key for cache-invalidation checks."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cache_is_valid(api_key: str) -> bool:
    """Return True if the cache exists, has not expired, and matches the key."""
    if not CACHE_FILE.exists():
        return False
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if data.get("api_key_hash") != _key_hash(api_key):
            return False
        return datetime.now() < datetime.fromisoformat(data["expires_at"])
    except Exception:
        return False


def load_cache() -> List[Dict[str, Any]]:
    """Return cached accessible models, or empty list if cache is absent/expired."""
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if datetime.now() < datetime.fromisoformat(data["expires_at"]):
            return data.get("models", [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Parameter size helper (for sorting results best-first)
# ---------------------------------------------------------------------------
def _param_billions(tag: str) -> float:
    import re
    clean = tag.replace("-cloud", "").replace(":cloud", "")
    moe = re.search(r"(\d+)[xX](\d+(?:\.\d+)?)b", clean, re.IGNORECASE)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    plain = re.search(r"(\d+(?:\.\d+)?)b", clean, re.IGNORECASE)
    return float(plain.group(1)) if plain else 0.0


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------
def run_probe(force: bool = False, verbose: bool = True) -> bool:
    """Run the full discovery + probe cycle.

    Parameters
    ----------
    force:
        When True the TTL is ignored and the cache is always regenerated.
    verbose:
        When True progress lines are printed to stdout.

    Returns
    -------
    bool
        True on success, False on any unrecoverable failure.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ── 1.  API key ──────────────────────────────────────────────────────────
    api_key = _get_api_key()
    if not api_key:
        _log(
            "\n[sentinel probe] OLLAMA_API_KEY is not configured.\n"
            "  Set it in ~/.sentinel/.env to enable cloud model discovery.\n"
            "  Generate a key at: https://ollama.com/settings/keys\n"
        )
        return False

    # ── 2.  TTL check ────────────────────────────────────────────────────────
    if not force and cache_is_valid(api_key):
        _log("[sentinel probe] Cache is current — nothing to do (use --force to override).")
        return True

    # ── 3.  Connectivity ─────────────────────────────────────────────────────
    try:
        import urllib.request
        urllib.request.urlopen(CLOUD_HOST, timeout=8)
    except Exception as exc:
        _log(f"[sentinel probe] Ollama Cloud unreachable ({exc}). Skipping — will retry next run.")
        return False

    # ── 4.  Import ollama SDK ────────────────────────────────────────────────
    try:
        from ollama import Client as _OllamaClient  # type: ignore[import]
    except ImportError:
        _log(
            "[sentinel probe] The 'ollama' Python package is missing.\n"
            "  Install it with:  pip install ollama"
        )
        return False

    client = _OllamaClient(
        host=CLOUD_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    # ── 5.  List all models ───────────────────────────────────────────────────
    _log("[sentinel probe] Fetching model catalogue from Ollama Cloud...")
    try:
        listing    = client.list()
        candidates = listing.get("models", [])
    except Exception as exc:
        _log(f"[sentinel probe] Could not list models: {exc}")
        return False

    if not candidates:
        _log("[sentinel probe] Model catalogue is empty — nothing to probe.")
        return False

    total = len(candidates)
    _log(f"[sentinel probe] Probing {total} model(s) with your API key...\n")

    # ── 6.  1-token probe per model ───────────────────────────────────────────
    accessible:   List[Dict[str, Any]] = []
    inaccessible: List[str]            = []

    for idx, m in enumerate(candidates, 1):
        tag = (m.get("model") or m.get("name") or m.get("id") or "").strip()
        if not tag:
            continue

        prefix = f"  [{idx:>2}/{total}]"
        try:
            client.generate(model=tag, prompt=" ", options=_PROBE_OPTS)
            accessible.append({
                "name":           tag,
                "cloud_tag":      to_cloud_tag(tag),
                "param_billions": _param_billions(tag),
                "probe_status":   "ok",
            })
            _log(f"{prefix}  ✓  {tag}")

        except Exception as exc:
            err = str(exc)
            is_auth_block = any(
                tok in err
                for tok in ("401", "402", "403", "Forbidden", "Unauthorized",
                            "Payment Required", "payment", "subscription")
            )
            if is_auth_block:
                inaccessible.append(tag)
                _log(f"{prefix}  ✗  {tag:<45} [plan restriction]")
            else:
                # Transient / unknown error — include optimistically
                accessible.append({
                    "name":           tag,
                    "cloud_tag":      to_cloud_tag(tag),
                    "param_billions": _param_billions(tag),
                    "probe_status":   "uncertain",
                })
                _log(f"{prefix}  ?  {tag:<45} [probe error — included tentatively]")

    # Sort best-first (largest parameter count first)
    accessible.sort(key=lambda x: x["param_billions"], reverse=True)

    # ── 7.  Write cache ───────────────────────────────────────────────────────
    SENTINEL_HOME.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    payload = {
        "version":      2,
        "generated_at": now.isoformat(),
        "expires_at":   (now + timedelta(hours=CACHE_TTL_HRS)).isoformat(),
        "api_key_hash": _key_hash(api_key),
        "models":       accessible,
        "inaccessible": inaccessible,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _log(
        f"\n[sentinel probe] Complete.\n"
        f"  Accessible : {len(accessible)}\n"
        f"  Blocked    : {len(inaccessible)}\n"
        f"  Cache      : {CACHE_FILE}\n"
        f"  Expires    : {payload['expires_at']}\n"
    )
    return len(accessible) > 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="cloud_model_probe",
        description="Probe Ollama Cloud to discover models accessible on your plan.",
    )
    ap.add_argument(
        "--force", "-f", action="store_true",
        help="Regenerate the cache even if it has not yet expired.",
    )
    ap.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress all output.",
    )
    ns = ap.parse_args()
    ok = run_probe(force=ns.force, verbose=not ns.quiet)
    sys.exit(0 if ok else 1)
