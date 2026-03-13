"""repo_explorer.py — Sentinel Repository Exploration Engine.

Runs a structured pre-planning exploration of the target project so that
all downstream agents (Planner, CodingAgent, etc.) operate with accurate
architectural knowledge rather than guessing from filenames alone.

Exploration pipeline
--------------------
::

    1. scan_structure()      → directory tree + file inventory
    2. detect_frameworks()   → language / framework / toolchain detection
    3. build_symbol_graph()  → class / function / import relationships (Python)
    4. generate_synopsis()   → LLM-written 300-word architectural summary
    5. produce_report()      → ExplorationReport dataclass

The report is cached under ``~/.sentinel/index/exploration_<hash>.json``
keyed on a SHA-256 of the project root mtime + size stats.  Subsequent
runs reuse the cache until source files change.

Framework detection
-------------------
Detection is heuristic — it examines ``package.json``, ``requirements.txt``,
``pyproject.toml``, ``Cargo.toml``, ``go.mod``, ``pom.xml``, and common
config files.  Detected stack example::

    {
        "languages": ["Python", "JavaScript"],
        "frontend":  "React",
        "backend":   "FastAPI",
        "database":  "PostgreSQL",
        "auth":      "JWT",
        "testing":   "pytest",
        "build":     "webpack",
        "container": "Docker"
    }

Integration
-----------
:class:`RepoExplorer` is called by
:class:`~agents.supervisor.ConcreteSupervisorAgent` immediately after it
parses the user prompt and before it calls the PlannerAgent.  The resulting
:class:`ExplorationReport` is stored on the context payload so every step
has access to it::

    context["exploration"] = report.to_dict()
    context["synopsis"]    = report.synopsis
    context["stack"]       = report.stack
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTINEL_HOME = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
_CACHE_DIR = _SENTINEL_HOME / "index"

_SKIP_DIRS: Set[str] = {
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    ".sentinel", ".idea", ".vscode", "dist", "build", ".next",
    "target", "bin", "obj", ".pytest_cache", ".mypy_cache",
}

_CODE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
}

_CONFIG_EXTENSIONS: Set[str] = {
    ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".env",
    ".lock", ".md", ".txt",
}

# Framework detection rules: (file_pattern, content_pattern, stack_key, value)
_DETECTION_RULES: List[Tuple[str, Optional[str], str, str]] = [
    # ── Frontend ──────────────────────────────────────────────────────
    ("package.json",       r'"react"',           "frontend",   "React"),
    ("package.json",       r'"vue"',             "frontend",   "Vue.js"),
    ("package.json",       r'"@angular/core"',   "frontend",   "Angular"),
    ("package.json",       r'"svelte"',          "frontend",   "Svelte"),
    ("package.json",       r'"next"',            "frontend",   "Next.js"),
    ("package.json",       r'"nuxt"',            "frontend",   "Nuxt.js"),
    # ── Backend ───────────────────────────────────────────────────────
    ("requirements.txt",   r'fastapi',           "backend",    "FastAPI"),
    ("requirements.txt",   r'flask',             "backend",    "Flask"),
    ("requirements.txt",   r'django',            "backend",    "Django"),
    ("requirements.txt",   r'tornado',           "backend",    "Tornado"),
    ("requirements.txt",   r'starlette',         "backend",    "Starlette"),
    ("package.json",       r'"express"',         "backend",    "Express"),
    ("package.json",       r'"fastify"',         "backend",    "Fastify"),
    ("package.json",       r'"koa"',             "backend",    "Koa"),
    ("pom.xml",            r'spring-boot',       "backend",    "Spring Boot"),
    ("go.mod",             r'gin-gonic/gin',     "backend",    "Gin"),
    ("go.mod",             r'labstack/echo',     "backend",    "Echo"),
    ("Cargo.toml",         r'actix-web',         "backend",    "Actix-web"),
    # ── Database ──────────────────────────────────────────────────────
    ("requirements.txt",   r'psycopg2|asyncpg',  "database",   "PostgreSQL"),
    ("requirements.txt",   r'pymysql|aiomysql',  "database",   "MySQL"),
    ("requirements.txt",   r'motor|pymongo',     "database",   "MongoDB"),
    ("requirements.txt",   r'redis',             "database",   "Redis"),
    ("requirements.txt",   r'sqlalchemy',        "database",   "SQLAlchemy"),
    ("package.json",       r'"mongoose"',        "database",   "MongoDB"),
    ("package.json",       r'"pg"',              "database",   "PostgreSQL"),
    ("package.json",       r'"mysql2"',          "database",   "MySQL"),
    ("package.json",       r'"redis"',           "database",   "Redis"),
    ("package.json",       r'"sequelize"',       "database",   "Sequelize"),
    # ── Auth ──────────────────────────────────────────────────────────
    ("requirements.txt",   r'python-jose|pyjwt', "auth",       "JWT"),
    ("requirements.txt",   r'authlib',           "auth",       "OAuth2"),
    ("requirements.txt",   r'passlib',           "auth",       "Passlib"),
    ("package.json",       r'"jsonwebtoken"',    "auth",       "JWT"),
    ("package.json",       r'"passport"',        "auth",       "Passport.js"),
    ("package.json",       r'"next-auth"',       "auth",       "NextAuth"),
    # ── Testing ───────────────────────────────────────────────────────
    ("requirements.txt",   r'pytest',            "testing",    "pytest"),
    ("requirements.txt",   r'unittest',          "testing",    "unittest"),
    ("package.json",       r'"jest"',            "testing",    "Jest"),
    ("package.json",       r'"mocha"',           "testing",    "Mocha"),
    ("package.json",       r'"vitest"',          "testing",    "Vitest"),
    # ── Build tools ───────────────────────────────────────────────────
    ("package.json",       r'"webpack"',         "build",      "Webpack"),
    ("package.json",       r'"vite"',            "build",      "Vite"),
    ("package.json",       r'"esbuild"',         "build",      "esbuild"),
    ("package.json",       r'"rollup"',          "build",      "Rollup"),
    ("Makefile",           None,                 "build",      "Make"),
    # ── Container / infra ─────────────────────────────────────────────
    ("Dockerfile",         None,                 "container",  "Docker"),
    ("docker-compose.yml", None,                 "container",  "Docker Compose"),
    ("docker-compose.yaml",None,                 "container",  "Docker Compose"),
    (".github/workflows",  None,                 "ci",         "GitHub Actions"),
    ("Jenkinsfile",        None,                 "ci",         "Jenkins"),
    (".gitlab-ci.yml",     None,                 "ci",         "GitLab CI"),
]

# Language detection by extension
_LANG_MAP: Dict[str, str] = {
    ".py":   "Python", ".js":  "JavaScript", ".ts": "TypeScript",
    ".jsx":  "React/JSX", ".tsx": "React/TSX",
    ".java": "Java",   ".go":  "Go",          ".rs": "Rust",
    ".cpp":  "C++",    ".c":   "C",           ".cs": "C#",
    ".rb":   "Ruby",   ".php": "PHP",         ".swift": "Swift",
    ".kt":   "Kotlin",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FileNode:
    """Lightweight metadata record for a single file in the project."""
    path: str           # project-root-relative path
    size_bytes: int
    extension: str
    language: Optional[str] = None
    is_entry_point: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationReport:
    """Full exploration report for a project.

    Attributes:
        project_root:   Absolute project root path.
        scanned_at:     Unix timestamp when exploration ran.
        total_files:    Total file count (all extensions).
        code_files:     Files with recognised code extensions.
        languages:      Detected programming languages (sorted by file count).
        stack:          Detected framework/toolchain stack dict.
        entry_points:   Likely entry-point file paths.
        top_dirs:       Top-level directory names (excluding skipped dirs).
        synopsis:       LLM-generated architectural summary (~300 words).
        symbol_summary: Brief class/function inventory string.
        from_cache:     True when loaded from disk cache.
        cache_key:      SHA-256 hash used for cache invalidation.
    """
    project_root: str
    scanned_at: float = field(default_factory=time.time)
    total_files: int = 0
    code_files: int = 0
    languages: List[str] = field(default_factory=list)
    stack: Dict[str, str] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    top_dirs: List[str] = field(default_factory=list)
    synopsis: str = ""
    symbol_summary: str = ""
    from_cache: bool = False
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_stack(self) -> str:
        """Return a human-readable stack summary string."""
        if not self.stack:
            return "Stack: unknown"
        lines = ["Detected stack:"]
        for key, val in self.stack.items():
            lines.append(f"  {key.capitalize()}: {val}")
        if self.languages:
            lines.append(f"  Languages: {', '.join(self.languages)}")
        return "\n".join(lines)

    def brief(self) -> str:
        """One-paragraph summary suitable for injecting into a system prompt."""
        stack_str = ", ".join(
            f"{k}={v}" for k, v in self.stack.items()
        ) if self.stack else "unknown stack"
        langs = ", ".join(self.languages) if self.languages else "unknown"
        eps = ", ".join(self.entry_points[:3]) if self.entry_points else "none detected"
        return (
            f"Project: {Path(self.project_root).name} | "
            f"Languages: {langs} | Stack: {stack_str} | "
            f"Entry points: {eps} | "
            f"Files: {self.total_files} total, {self.code_files} code files."
        )


# ---------------------------------------------------------------------------
# RepoExplorer
# ---------------------------------------------------------------------------


class RepoExplorer:
    """Explores a repository and produces an :class:`ExplorationReport`.

    Args:
        project_root:   Absolute path to the project directory.
        ollama_client:  Optional :class:`~models.ollama_client.OllamaClient`
                        for LLM synopsis generation.
        synopsis_model: Ollama model tag for synopsis generation.
        use_cache:      Whether to read/write the disk cache (default True).
        max_files:      Maximum files to scan (default 2000).
    """

    def __init__(
        self,
        project_root: str,
        ollama_client: Optional[Any] = None,
        synopsis_model: str = "mistral:7b",
        use_cache: bool = True,
        max_files: int = 2000,
    ) -> None:
        self.root = Path(project_root).resolve()
        self._ollama = ollama_client
        self._synopsis_model = synopsis_model
        self._use_cache = use_cache
        self._max_files = max_files
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: explore
    # ------------------------------------------------------------------

    def explore(self, force: bool = False) -> ExplorationReport:
        """Run the full exploration pipeline and return a report.

        Args:
            force: Bypass the cache and re-run exploration.

        Returns:
            :class:`ExplorationReport` — either freshly generated or
            restored from the disk cache.
        """
        cache_key = self._compute_cache_key()

        if not force and self._use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # ── 1. Scan structure ────────────────────────────────────────
        files = self._scan_structure()

        # ── 2. Detect frameworks ─────────────────────────────────────
        stack = self._detect_frameworks()
        languages = self._detect_languages(files)
        entry_points = self._detect_entry_points(files)
        top_dirs = self._top_dirs()

        # ── 3. Symbol summary ────────────────────────────────────────
        symbol_summary = self._build_symbol_summary(files)

        # ── 4. LLM synopsis ──────────────────────────────────────────
        synopsis = self._generate_synopsis(files, stack, languages)

        report = ExplorationReport(
            project_root=str(self.root),
            scanned_at=time.time(),
            total_files=len(files),
            code_files=sum(1 for f in files if f.language),
            languages=languages,
            stack=stack,
            entry_points=entry_points,
            top_dirs=top_dirs,
            synopsis=synopsis,
            symbol_summary=symbol_summary,
            from_cache=False,
            cache_key=cache_key,
        )

        if self._use_cache:
            self._save_cache(cache_key, report)

        return report

    # ------------------------------------------------------------------
    # 1. Structure scan
    # ------------------------------------------------------------------

    def _scan_structure(self) -> List[FileNode]:
        """Walk the project tree and return a list of :class:`FileNode` records."""
        files: List[FileNode] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune skipped directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                full = Path(dirpath) / fname
                try:
                    rel = str(full.relative_to(self.root))
                    size = full.stat().st_size
                    ext = full.suffix.lower()
                    lang = _LANG_MAP.get(ext)
                    files.append(FileNode(
                        path=rel,
                        size_bytes=size,
                        extension=ext,
                        language=lang,
                    ))
                except (OSError, ValueError):
                    continue
                if len(files) >= self._max_files:
                    return files
        return files

    # ------------------------------------------------------------------
    # 2. Framework detection
    # ------------------------------------------------------------------

    def _detect_frameworks(self) -> Dict[str, str]:
        """Scan config files and apply detection rules."""
        stack: Dict[str, str] = {}
        for file_pattern, content_pattern, stack_key, value in _DETECTION_RULES:
            if stack_key in stack:
                continue  # Already detected this slot
            candidate = self.root / file_pattern
            if candidate.exists():
                if content_pattern is None:
                    stack[stack_key] = value
                else:
                    try:
                        text = candidate.read_text(encoding="utf-8", errors="ignore")
                        if re.search(content_pattern, text, re.IGNORECASE):
                            stack[stack_key] = value
                    except OSError:
                        continue
        return stack

    def _detect_languages(self, files: List[FileNode]) -> List[str]:
        """Return languages sorted by file count (most common first)."""
        counts: Dict[str, int] = {}
        for f in files:
            if f.language:
                counts[f.language] = counts.get(f.language, 0) + 1
        return [lang for lang, _ in sorted(counts.items(), key=lambda x: -x[1])]

    def _detect_entry_points(self, files: List[FileNode]) -> List[str]:
        """Identify likely entry-point files by name heuristics."""
        entry_names = {
            "main.py", "__main__.py", "app.py", "cli.py", "server.py",
            "index.js", "index.ts", "app.js", "app.ts", "server.js",
            "main.go", "main.rs", "Main.java", "Program.cs",
        }
        results: List[str] = []
        for f in files:
            if Path(f.path).name in entry_names:
                results.append(f.path)
        # Also catch cli*.py patterns
        for f in files:
            name = Path(f.path).name
            if re.match(r"cli.*\.py", name, re.IGNORECASE) and f.path not in results:
                results.append(f.path)
        return results[:6]

    def _top_dirs(self) -> List[str]:
        """Return names of top-level non-skipped directories."""
        try:
            return [
                d.name for d in sorted(self.root.iterdir())
                if d.is_dir()
                and d.name not in _SKIP_DIRS
                and not d.name.startswith(".")
            ]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # 3. Symbol summary
    # ------------------------------------------------------------------

    def _build_symbol_summary(self, files: List[FileNode]) -> str:
        """Build a concise inventory of classes and top-level functions.

        Scans up to 20 Python files using the stdlib ``ast`` module.
        Returns a multi-line string like::

            core/engine.py: ConcreteExecutionEngine, run_pipeline, _run_solo
            agents/coding_agent.py: CodingAgent, _llm_actions, _parse_llm_actions
        """
        import ast as _ast

        py_files = [f for f in files if f.extension == ".py"][:20]
        lines: List[str] = []
        for fnode in py_files:
            full = self.root / fnode.path
            try:
                src = full.read_text(encoding="utf-8", errors="ignore")
                tree = _ast.parse(src, filename=fnode.path)
            except Exception:
                continue
            symbols: List[str] = []
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                    if not isinstance(getattr(node, "parent", None), _ast.ClassDef):
                        symbols.append(node.name)
            if symbols:
                lines.append(f"{fnode.path}: {', '.join(symbols[:8])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 4. LLM synopsis
    # ------------------------------------------------------------------

    def _generate_synopsis(
        self,
        files: List[FileNode],
        stack: Dict[str, str],
        languages: List[str],
    ) -> str:
        """Generate a ~300-word architectural synopsis using the LLM.

        Selects representative files (README, entry points, largest code
        files) and sends them to the Ollama model.  Falls back to a
        structured text summary when Ollama is unavailable.
        """
        if self._ollama is None:
            return self._fallback_synopsis(stack, languages, files)

        sample = self._select_sample_files(files)
        if not sample:
            return self._fallback_synopsis(stack, languages, files)

        file_sections: List[str] = []
        for path in sample:
            full = self.root / path
            try:
                text = full.read_text(encoding="utf-8", errors="ignore")[:1500]
                file_sections.append(f"### {path}\n```\n{text}\n```")
            except OSError:
                continue

        stack_hint = (
            "Detected stack: " + ", ".join(f"{k}={v}" for k, v in stack.items())
            if stack else ""
        )

        prompt = (
            "You are an expert software architect. Read these source files and write "
            "a concise technical summary (~300 words) covering: "
            "(1) what the project does, "
            "(2) its main components and how they interact, "
            "(3) key entry points and important classes or modules. "
            "Be precise and avoid filler language.\n\n"
            + (stack_hint + "\n\n" if stack_hint else "")
            + "\n\n".join(file_sections)
        )

        try:
            response = self._ollama.generate(
                model=self._synopsis_model,
                prompt=prompt,
                options={"num_predict": 512, "temperature": 0.1},
                timeout=120,
            )
            text = response.get("response", "").strip()
            return text if text else self._fallback_synopsis(stack, languages, files)
        except Exception:
            return self._fallback_synopsis(stack, languages, files)

    def _fallback_synopsis(
        self,
        stack: Dict[str, str],
        languages: List[str],
        files: List[FileNode],
    ) -> str:
        """Produce a structured text synopsis when LLM is unavailable."""
        lang_str = ", ".join(languages[:4]) if languages else "unknown"
        stack_str = (
            "; ".join(f"{k}: {v}" for k, v in stack.items())
            if stack else "not detected"
        )
        code_count = sum(1 for f in files if f.language)
        return (
            f"Project: {self.root.name}\n"
            f"Languages: {lang_str}\n"
            f"Stack: {stack_str}\n"
            f"Code files: {code_count}\n"
            "(LLM synopsis unavailable — Ollama not configured)"
        )

    def _select_sample_files(self, files: List[FileNode]) -> List[str]:
        """Choose ≤10 representative files for the synopsis prompt."""
        selected: List[str] = []
        seen: Set[str] = set()

        def _add(p: str) -> None:
            if p not in seen:
                seen.add(p)
                selected.append(p)

        # Prefer README first
        for fname in ("README.md", "README.rst", "README.txt"):
            if (self.root / fname).exists():
                _add(fname)

        # Entry points
        for p in self._detect_entry_points(files):
            _add(p)

        # Largest code files
        code = sorted(
            [f for f in files if f.language and f.path not in seen],
            key=lambda f: -f.size_bytes,
        )
        for f in code:
            if len(selected) >= 10:
                break
            _add(f.path)

        return selected[:10]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _compute_cache_key(self) -> str:
        """Hash project root + mtime of top-level files for cache invalidation."""
        h = hashlib.sha256()
        h.update(str(self.root).encode())
        try:
            for child in sorted(self.root.iterdir())[:30]:
                try:
                    stat = child.stat()
                    h.update(f"{child.name}:{stat.st_mtime}:{stat.st_size}".encode())
                except OSError:
                    pass
        except OSError:
            pass
        return h.hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return _CACHE_DIR / f"exploration_{key}.json"

    def _load_cache(self, key: str) -> Optional[ExplorationReport]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            r = ExplorationReport(**{
                k: v for k, v in data.items()
                if k in ExplorationReport.__dataclass_fields__
            })
            r.from_cache = True
            return r
        except Exception:
            return None

    def _save_cache(self, key: str, report: ExplorationReport) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(report.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
