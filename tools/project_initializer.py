"""project_initializer.py — Sentinel ProjectInitializer tool.

Detects the project type from a goal/description and runs the appropriate
scaffolding command (e.g. ``npx create-react-app``, ``npm init``,
``flutter create``, etc.) **inside the output_dir itself** — no extra
subdirectory is created.  All CLIs are invoked with either ``.`` as the
target path or equivalent flags so the project structure lands directly
in ``output_dir``.

Supported project types
-----------------------
Web / Frontend
  react          — npx create-react-app .
  react-ts       — npx create-react-app . --template typescript
  vite           — npm create vite@latest . (react template)
  nextjs         — npx create-next-app@latest .
  vue            — npm create vue@latest .
  angular        — npx @angular/cli new <name> --directory .
  svelte         — npm create svelte@latest .

Backend / Node
  node           — npm init -y
  express        — npm init -y + npm install express
  fastify        — npm init -y + npm install fastify

Backend / Python
  python         — python -m venv venv
  fastapi        — python -m venv venv + pip install fastapi uvicorn
  django         — python -m venv venv + django-admin startproject <name> .
  flask          — python -m venv venv + pip install flask

Mobile
  react-native   — npx react-native@latest init <name> --directory .
  expo           — npx create-expo-app@latest .
  flutter        — flutter create .
  kotlin-android — gradle init (in-place)
  swift-ios      — swift package init (in-place)

Desktop / Game
  unity          — Warns user to use Unity Hub CLI
  unreal         — Warns user to use Unreal Engine CLI
  godot          — Warns user to use Godot editor
  tauri          — npm create tauri-app@latest .
  electron       — npm init + npm install electron

Registered name: ``"project_initializer"``
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from tools.tool_registry import Tool, ToolResult

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------
_SHORT_TIMEOUT = 30    # seconds — for quick installs / checks
_LONG_TIMEOUT  = 300   # seconds — for scaffold commands that download packages

# ---------------------------------------------------------------------------
# Project-type detection keyword map
# Each entry: (project_type, weight, keywords)
# ---------------------------------------------------------------------------
_DETECTION_MATRIX: List[Tuple[str, float, List[str]]] = [
    # Mobile
    ("flutter",        3.0, ["flutter", "flutter app", "flutter project"]),
    ("expo",           3.0, ["expo", "expo app", "expo react native"]),
    ("react-native",   2.5, ["react native", "react-native", "rn app"]),
    ("kotlin-android", 2.5, ["kotlin", "android kotlin", "android app", "kotlin android"]),
    ("swift-ios",      2.5, ["swift", "ios app", "swiftui", "xcode project"]),
    # Game engines
    ("unity",          3.0, ["unity", "unity3d", "unity game", "unity project"]),
    ("unreal",         3.0, ["unreal", "ue5", "ue4", "unreal engine"]),
    ("godot",          3.0, ["godot", "godot game", "gdscript"]),
    # Desktop
    ("tauri",          2.5, ["tauri", "tauri app", "tauri desktop"]),
    ("electron",       2.5, ["electron", "electron app", "electron desktop"]),
    # Web frameworks (order matters — more specific first)
    ("react-ts",       2.5, ["react typescript", "react ts", "create react app typescript"]),
    ("nextjs",         2.5, ["next.js", "nextjs", "next js", "next app"]),
    ("angular",        2.5, ["angular", "angular app", "ng new"]),
    ("svelte",         2.5, ["svelte", "sveltekit", "svelte app"]),
    ("vue",            2.5, ["vue", "vue.js", "vuejs", "vue app"]),
    ("vite",           2.0, ["vite", "vite app", "vite project"]),
    ("react",          2.0, ["react", "create react app", "react app", "react website"]),
    # Backend Python
    ("django",         2.5, ["django", "django project", "django app"]),
    ("fastapi",        2.5, ["fastapi", "fast api"]),
    ("flask",          2.5, ["flask", "flask app", "flask api"]),
    ("python",         1.0, ["python project", "python backend", "python script", "python app"]),
    # Backend Node
    ("fastify",        2.5, ["fastify"]),
    ("express",        2.0, ["express", "expressjs", "express api", "express server", "node express"]),
    ("node",           1.0, ["node.js", "nodejs", "node backend", "node server", "npm project"]),
]

# ---------------------------------------------------------------------------
# Scaffold command recipes
# Each entry maps project_type -> list of shell command strings
# Commands may reference {name} and {path} placeholders.
# ---------------------------------------------------------------------------
_SCAFFOLD_RECIPES: Dict[str, List[str]] = {
    # ── Web / Frontend ───────────────────────────────────────────────────────
    # All CLIs are invoked with "." so the scaffold lands in the current dir.
    "react": [
        "npx --yes create-react-app .",
    ],
    "react-ts": [
        "npx --yes create-react-app . --template typescript",
    ],
    "vite": [
        # --overwrite silences "target directory is not empty" when CI=true
        # would otherwise cause create-vite to abort instead of proceeding.
        "npm create --yes vite@latest . -- --template react --overwrite",
        "npm install",
    ],
    "nextjs": [
        "npx --yes create-next-app@latest . --no-interactive",
    ],
    "vue": [
        # Same --overwrite requirement as vite (uses the same create-vue CLI)
        "npm create --yes vue@latest . -- --overwrite",
        "npm install",
    ],
    "angular": [
        # --directory . places the generated files into the current directory
        # while still using {name} as the Angular project/app identifier.
        "npx --yes @angular/cli new {name} --directory . --skip-git --skip-install",
        "npm install",
    ],
    "svelte": [
        # SvelteKit's create CLI also prompts on non-empty dirs; --force skips it
        "npm create --yes svelte@latest . -- --force",
        "npm install",
    ],
    # ── Backend / Node ───────────────────────────────────────────────────────
    "node": [
        "npm init -y",
    ],
    "express": [
        "npm init -y",
        "npm install express",
    ],
    "fastify": [
        "npm init -y",
        "npm install fastify",
    ],
    # ── Backend / Python ─────────────────────────────────────────────────────
    "python": [
        "python -m venv venv",
    ],
    "fastapi": [
        "python -m venv venv",
        "{pip} install fastapi uvicorn[standard]",
    ],
    "django": [
        # Trailing "." tells django-admin to create manage.py + the inner
        # package directly in the current directory instead of a subdirectory.
        "python -m venv venv",
        "{pip} install django",
        "{django_admin} startproject {name} .",
    ],
    "flask": [
        "python -m venv venv",
        "{pip} install flask",
    ],
    # ── Mobile ───────────────────────────────────────────────────────────────
    "react-native": [
        # --directory . initialises in the current directory; the project
        # identifier {name} is still required by the RN CLI.
        "npx --yes react-native@latest init {name} --directory .",
    ],
    "expo": [
        "npx --yes create-expo-app@latest .",
    ],
    "flutter": [
        "flutter create .",
    ],
    "kotlin-android": [
        "gradle init --type basic --dsl kotlin --project-name {name} --no-incubating",
    ],
    "swift-ios": [
        "swift package init --name {name} --type executable",
    ],
    # ── Desktop ──────────────────────────────────────────────────────────────
    "electron": [
        "npm init -y",
        "npm install --save-dev electron",
    ],
    "tauri": [
        "npm create --yes tauri-app@latest .",
    ],
    # ── Game Engines (require GUI / proprietary CLI — emit guidance only) ────
    "unity":  [],   # handled separately with a user-facing message
    "unreal": [],
    "godot":  [],
}

# ---------------------------------------------------------------------------
# Type aliases — common LLM-generated type strings that aren't valid recipe
# keys but map clearly onto a canonical type.  Applied before detection so
# the LLM can say "web" or "website" and get a sensible scaffold.
# ---------------------------------------------------------------------------
_TYPE_ALIASES: Dict[str, str] = {
    # Generic web / frontend
    "web":          "react",
    "website":      "react",
    "webpage":      "react",
    "webapp":       "react",
    "frontend":     "react",
    "html":         "react",
    "static":       "react",
    "spa":          "react",
    "typescript":   "react-ts",
    "ts":           "react-ts",
    "next":         "nextjs",
    "next.js":      "nextjs",
    "nuxt":         "vue",
    "nuxt.js":      "vue",
    "angular-app":  "angular",
    "ng":           "angular",
    "sveltekit":    "svelte",
    # Backend
    "backend":      "node",
    "server":       "node",
    "api":          "express",
    "rest":         "express",
    "rest-api":     "express",
    "js":           "node",
    "javascript":   "node",
    "node.js":      "node",
    "nodejs":       "node",
    "py":           "python",
    "script":       "python",
    "fast-api":     "fastapi",
    # Mobile
    "rn":           "react-native",
    "reactnative":  "react-native",
    "android":      "kotlin-android",
    "ios":          "swift-ios",
    "xcode":        "swift-ios",
    # Desktop
    "desktop":      "electron",
    "cross-platform": "electron",
}

# Human-readable guidance for IDE/GUI-only engines
_ENGINE_GUIDANCE: Dict[str, str] = {
    "unity": (
        "Unity projects must be initialised through the Unity Hub or Unity Editor. "
        "Steps:\n"
        "  1. Open Unity Hub → Projects → New Project.\n"
        "  2. Select your template (3D, 2D, URP, HDRP, etc.).\n"
        "  3. Set the project name and location, then click Create.\n"
        "Alternatively, use the Unity Hub CLI (if installed):\n"
        "  unity-hub -- --headless create-project --projectPath /path/to/{name}\n"
        "Once the project exists, the assistant can generate scripts and assets inside it."
    ),
    "unreal": (
        "Unreal Engine projects must be created via the Epic Games Launcher or the "
        "UnrealEditor CLI. Steps:\n"
        "  1. Open Epic Games Launcher → Unreal Engine → Launch.\n"
        "  2. Games → New Project → choose a template (Blank, First Person, etc.).\n"
        "  3. Set language (C++ or Blueprint), project name and location.\n"
        "For headless creation (UE5+):\n"
        "  UnrealEditor-Cmd.exe -createproject ProjectName=/path/to/{name} "
        "-projecttemplate=TP_Blank_BP -nodev\n"
        "Once the project exists, the assistant can generate C++ source files and Blueprints."
    ),
    "godot": (
        "Godot projects are initialised through the Godot editor or by creating a "
        "minimal project.godot file manually. Steps:\n"
        "  1. Open Godot → Project Manager → New Project.\n"
        "  2. Set the name, path and renderer, then click Create.\n"
        "Headless creation:\n"
        "  godot --headless --path /path/to/{name} --quit  (creates project.godot)\n"
        "Once the project exists, the assistant can generate GDScript / C# source files."
    ),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_project_type(description: str) -> Optional[str]:
    """Score *description* against the detection matrix and return best match."""
    text = description.lower()
    scores: Dict[str, float] = {}
    for proj_type, weight, keywords in _DETECTION_MATRIX:
        for kw in keywords:
            if kw in text:
                scores[proj_type] = scores.get(proj_type, 0.0) + weight
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------

def _sanitize_npm_name(name: str) -> str:
    """Convert any string into a valid npm package / directory name.

    npm naming rules:
    * All lowercase
    * URL-safe characters only (letters, digits, hyphens, dots, underscores)
    * Cannot start with a dot or underscore
    * No spaces; no capital letters; total length ≤ 214 characters

    Examples
    --------
    >>> _sanitize_npm_name("Hello world webpage")
    'hello-world-webpage'
    >>> _sanitize_npm_name("My App v2.0!")
    'my-app-v2.0'
    """
    name = name.lower().strip()
    # Replace anything that is not a-z, 0-9, hyphen, dot, or underscore with a hyphen
    name = re.sub(r"[^a-z0-9.\-_]", "-", name)
    # Collapse consecutive hyphens / dots
    name = re.sub(r"-+", "-", name)
    name = re.sub(r"\.+", ".", name)
    # Strip leading/trailing hyphens, dots, underscores
    name = name.strip("-._")
    # Must not start with a dot or underscore
    name = re.sub(r"^[._]+", "", name)
    # Enforce length limit
    name = name[:214]
    return name or "my-project"


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_FRONTEND_TYPES: frozenset = frozenset({
    "react", "react-ts", "vite", "nextjs", "vue", "angular", "svelte",
    "react-native", "expo", "flutter", "kotlin-android", "swift-ios",
    "electron", "tauri",
})

_BACKEND_TYPES: frozenset = frozenset({
    "node", "express", "fastify", "python", "fastapi", "django", "flask",
})

# Regex patterns that strongly indicate a fullstack project description
_FULLSTACK_PATTERNS: List[str] = [
    r"full[\s\-]?stack",
    r"front[\s\-]?end\b.{0,40}\bback[\s\-]?end",
    r"back[\s\-]?end\b.{0,40}\bfront[\s\-]?end",
    r"client\b.{0,30}\bserver",
    r"server\b.{0,30}\bclient",
    r"(react|vue|angular|svelte|next(?:js)?)\b.{0,50}\b(api|backend|express|django|fastapi|flask|node)",
    r"(express|django|fastapi|flask|node(?:js)?)\b.{0,50}\b(react|vue|angular|svelte|next(?:js)?|frontend)",
    r"\bapi\b.{0,30}\b(ui|interface|dashboard|frontend)",
    r"\b(ui|interface|dashboard|frontend)\b.{0,30}\bapi\b",
]


def _detect_architecture(project_type: str, description: str) -> str:
    """Return ``'frontend'``, ``'backend'``, or ``'fullstack'``.

    Checks *description* for fullstack signals first, then falls back to
    inferring from *project_type*.  Returns ``'frontend'`` as a safe default.
    """
    text = (description or "").lower()
    for pattern in _FULLSTACK_PATTERNS:
        if re.search(pattern, text):
            return "fullstack"
    if project_type in _BACKEND_TYPES:
        return "backend"
    return "frontend"


# ---------------------------------------------------------------------------
# Shared scaffold runner (used for both single-arch and each half of fullstack)
# ---------------------------------------------------------------------------

def _scaffold_dir(
    project_type: str,
    safe_name: str,
    cwd: str,
    timeout: int,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Run the scaffold recipe for *project_type* in *cwd*.

    Returns ``(step_results, all_ok)``.
    """
    commands = _SCAFFOLD_RECIPES.get(project_type, [])
    if not commands:
        return (
            [{"command": "", "success": False, "stdout": "",
              "stderr": f"No scaffold recipe defined for project type '{project_type}'."}],
            False,
        )

    pip = _pip_executable(cwd)
    django_admin = _django_admin_executable(cwd)
    step_results: List[Dict[str, Any]] = []
    all_ok = True

    for raw_cmd in commands:
        cmd = raw_cmd.format(
            name=safe_name,
            path=cwd,
            pip=pip,
            django_admin=django_admin,
        )
        if progress_callback is not None:
            try:
                progress_callback(f"$ {cmd}")
            except Exception:
                pass
        pip = _pip_executable(cwd)
        django_admin = _django_admin_executable(cwd)
        ok, stdout, stderr = _run_command(cmd, cwd=cwd, timeout=timeout, progress_callback=progress_callback)
        step_results.append({
            "command": cmd,
            "success": ok,
            "stdout": stdout[:4000],
            "stderr": stderr[:2000],
        })
        if not ok:
            all_ok = False
            break

    return step_results, all_ok


# ---------------------------------------------------------------------------
# Post-scaffold helpers
# ---------------------------------------------------------------------------

def _patch_package_json(directory: str, safe_name: str) -> None:
    """Overwrite the ``name`` field in *directory*/package.json with *safe_name*.

    npm-based scaffold tools (create-react-app, vite, etc.) derive the
    package name from the directory name.  When the directory name contains
    spaces or capital letters, ``npm`` rejects it.  This function patches the
    generated ``package.json`` after scaffolding so the name is always valid.
    """
    import json as _json
    pkg_path = os.path.join(directory, "package.json")
    if not os.path.isfile(pkg_path):
        return
    try:
        with open(pkg_path, "r", encoding="utf-8") as fh:
            data = _json.load(fh)
        data["name"] = safe_name
        with open(pkg_path, "w", encoding="utf-8") as fh:
            _json.dump(data, fh, indent=2)
    except Exception:  # noqa: BLE001 — non-critical
        pass


def _pip_executable(cwd: str) -> str:
    """Return the pip path inside the project's venv if it exists.

    The venv is expected at ``<cwd>/venv`` — created directly in the
    project directory now that scaffolds run in-place.
    """
    venv_dir = os.path.join(cwd, "venv")
    if sys.platform == "win32":
        candidate = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        candidate = os.path.join(venv_dir, "bin", "pip")
    return candidate if os.path.isfile(candidate) else "pip"


def _django_admin_executable(cwd: str) -> str:
    """Return the django-admin path inside the project's venv if it exists."""
    venv_dir = os.path.join(cwd, "venv")
    if sys.platform == "win32":
        candidate = os.path.join(venv_dir, "Scripts", "django-admin.exe")
    else:
        candidate = os.path.join(venv_dir, "bin", "django-admin")
    return candidate if os.path.isfile(candidate) else "django-admin"


def _run_command(
    command: str,
    cwd: str,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[bool, str, str]:
    """Run a shell command and return (success, stdout, stderr).

    When *progress_callback* is provided it is called with each stdout line
    as it arrives (using ``subprocess.Popen`` + line iteration instead of
    ``subprocess.run`` so the callback fires in real-time).
    """
    _env = dict(os.environ)
    # Suppress interactive prompts
    _env["CI"] = "true"
    _env["DEBIAN_FRONTEND"] = "noninteractive"
    if env:
        _env.update(env)

    try:
        if progress_callback is not None:
            # Streaming variant: call callback for each stdout line
            import io as _io
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_lines = []
            stderr_data = b""
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    stdout_lines.append(line)
                    try:
                        progress_callback(line)
                    except Exception:
                        pass
                proc.wait(timeout=timeout)
                if proc.stderr:
                    stderr_data = proc.stderr.read()
            except subprocess.TimeoutExpired:
                proc.kill()
                return False, "", f"Command timed out after {timeout}s: {command}"
            stdout = "\n".join(stdout_lines)
            stderr = stderr_data.decode("utf-8", errors="replace")
            return proc.returncode == 0, stdout, stderr
        else:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=_env,
                capture_output=True,
                timeout=timeout,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            return result.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s: {command}"
    except Exception as exc:  # noqa: BLE001
        return False, "", str(exc)


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class ProjectInitializerTool(Tool):
    """Scaffold a new project using the appropriate CLI initialiser.

    Parameters
    ----------
    project_name : str
        The name / slug for the new project directory.
    project_type : str, optional
        Explicit project type (e.g. ``"react"``, ``"flutter"``).
        When omitted, the tool infers the type from *description*.
    description : str, optional
        Free-text goal description used for auto-detection when
        *project_type* is not supplied.
    output_dir : str, optional
        Parent directory in which to create the project.
        Defaults to the current working directory (``"."``).
    timeout : int, optional
        Per-command timeout in seconds.  Defaults to 300.
    """

    name = "project_initializer"
    description = (
        "Scaffold a new project by running the appropriate CLI initialiser "
        "(e.g. npx create-react-app, flutter create, npm init, django-admin startproject) "
        "DIRECTLY INSIDE output_dir — no extra subdirectory is created. "
        "Call this BEFORE writing any source files for a brand-new project. "
        "Common aliases are resolved automatically: 'web'→react, 'website'→react, "
        "'api'→express, 'backend'→node, 'py'→python, 'rn'→react-native, etc."
    )
    parameters_schema = {
        "project_name": {
            "type": "string",
            "description": "Name / identifier for the project (used by CLIs that require one, e.g. Angular, Django, React Native).",
            "required": True,
        },
        "project_type": {
            "type": "string",
            "description": (
                "Explicit project type. Canonical values: react, react-ts, vite, nextjs, vue, "
                "angular, svelte, node, express, fastify, python, fastapi, django, flask, "
                "react-native, expo, flutter, kotlin-android, swift-ios, "
                "electron, tauri, unity, unreal, godot. "
                "Common aliases also accepted: web, website, webapp, frontend, html, static, "
                "api, backend, server, js, ts, typescript, py, rn, android, ios, desktop, etc. "
                "Leave blank to auto-detect from description."
            ),
            "required": False,
            "default": "",
        },
        "description": {
            "type": "string",
            "description": "Goal / task description — used to auto-detect project type.",
            "required": False,
            "default": "",
        },
        "output_dir": {
            "type": "string",
            "description": (
                "Directory in which to initialise the project. "
                "The scaffold runs HERE — no subdirectory is created inside it. "
                "Should be the project_root. Defaults to '.'."
            ),
            "required": False,
            "default": ".",
        },
        "architecture": {
            "type": "string",
            "description": (
                "Project architecture. One of: 'frontend', 'backend', 'fullstack'. "
                "Use 'fullstack' when the project needs both a UI layer and an API/server — "
                "this creates frontend/ and backend/ subdirectories inside output_dir. "
                "Leave blank for auto-detection from description and project_type."
            ),
            "required": False,
            "default": "",
        },
        "frontend_type": {
            "type": "string",
            "description": (
                "Frontend framework for fullstack projects "
                "(e.g. 'react', 'vue', 'nextjs', 'svelte'). "
                "Only used when architecture='fullstack'."
            ),
            "required": False,
            "default": "",
        },
        "backend_type": {
            "type": "string",
            "description": (
                "Backend framework for fullstack projects "
                "(e.g. 'express', 'fastapi', 'django', 'flask'). "
                "Only used when architecture='fullstack'."
            ),
            "required": False,
            "default": "",
        },
        "timeout": {
            "type": "int",
            "description": "Per-command timeout in seconds. Defaults to 300.",
            "required": False,
            "default": _LONG_TIMEOUT,
        },
    }

    def run(  # type: ignore[override]
        self,
        project_name: str,
        project_type: str = "",
        description: str = "",
        output_dir: str = ".",
        architecture: str = "",
        frontend_type: str = "",
        backend_type: str = "",
        timeout: int = _LONG_TIMEOUT,
        progress_callback: Optional[Any] = None,
        **_: Any,
    ) -> ToolResult:
        # Store callback for use in _run_command calls
        self._progress_callback = progress_callback
        # ── 1. Sanitise project name to npm / filesystem naming rules ─────────
        safe_name = _sanitize_npm_name(project_name) or "my-project"

        # ── 2. Resolve / create output directory ──────────────────────────────
        # IMPORTANT: npm-based scaffolders derive the package name from the
        # directory they run in.  We must ensure the basename is npm-safe
        # BEFORE creating the directory — a post-hoc os.rename() is unreliable
        # on Windows (the OS may refuse to rename a directory that Explorer or
        # another process has open, and the silent OSError fallback means npm
        # then runs in the unsafe-named directory and fails with a
        # "naming restrictions" error).
        #
        # Strategy: compute the safe basename first, substitute it into the
        # resolved path, then makedirs.  `renamed_dir` is set so callers
        # (e.g. the execution engine) learn the actual path used.
        _raw_out      = os.path.realpath(output_dir)
        dir_basename  = os.path.basename(_raw_out)
        safe_dir_name = _sanitize_npm_name(dir_basename)
        renamed_dir: Optional[str] = None

        if safe_dir_name and safe_dir_name != dir_basename:
            abs_out     = os.path.join(os.path.dirname(_raw_out), safe_dir_name)
            renamed_dir = abs_out
        else:
            abs_out = _raw_out

        os.makedirs(abs_out, exist_ok=True)

        # ── 3. Detect and normalise project type ──────────────────────────────
        detected_type = project_type.strip().lower() if project_type.strip() else None
        if not detected_type:
            detected_type = _detect_project_type(description or "")
        if not detected_type:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "Could not determine project type. "
                    "Please pass 'project_type' explicitly. "
                    f"Supported types: {', '.join(sorted(_SCAFFOLD_RECIPES))}."
                ),
            )
        detected_type = _TYPE_ALIASES.get(detected_type, detected_type)

        # ── 4. Detect architecture ────────────────────────────────────────────
        arch = (architecture or "").strip().lower()
        if arch not in ("frontend", "backend", "fullstack"):
            arch = _detect_architecture(detected_type, description or "")

        # ── 5. Fullstack path — scaffold frontend/ and backend/ separately ────
        if arch == "fullstack":
            # Resolve frontend/backend framework types
            _fe = _TYPE_ALIASES.get(
                frontend_type.strip().lower(), frontend_type.strip().lower()
            ) or (detected_type if detected_type in _FRONTEND_TYPES else "react")
            _be = _TYPE_ALIASES.get(
                backend_type.strip().lower(), backend_type.strip().lower()
            ) or (detected_type if detected_type in _BACKEND_TYPES else "express")
            return self._run_fullstack(
                safe_name=safe_name, frontend_type=_fe, backend_type=_be,
                abs_out=abs_out, dir_basename=dir_basename,
                renamed_dir=renamed_dir, timeout=timeout,
                progress_callback=getattr(self, "_progress_callback", None),
            )

        # ── 6. Single-arch path ───────────────────────────────────────────────
        if detected_type in _ENGINE_GUIDANCE:
            guidance = _ENGINE_GUIDANCE[detected_type].format(name=safe_name)
            return ToolResult(
                tool_name=self.name, success=True,
                output={
                    "architecture": arch, "project_type": detected_type,
                    "project_name": safe_name, "output_dir": abs_out,
                    "initialized": False, "guidance": guidance,
                    "steps": [], "renamed_dir": renamed_dir,
                    "message": (
                        f"{detected_type.capitalize()} projects cannot be initialised "
                        "from the command line without the engine installed. "
                        "See 'guidance' for manual steps."
                    ),
                },
            )

        if detected_type not in _SCAFFOLD_RECIPES:
            _aliases_hint = ", ".join(
                f"'{k}' → {v}" for k, v in sorted(_TYPE_ALIASES.items())
                if v == detected_type or k == detected_type
            )
            return ToolResult(
                tool_name=self.name, success=False,
                error=(
                    f"Unknown project type '{detected_type}'. "
                    f"Supported: {', '.join(sorted(_SCAFFOLD_RECIPES))}. "
                    + (f"Known aliases: {_aliases_hint}. " if _aliases_hint else "")
                    + "Common aliases: 'web'→react, 'website'→react, 'api'→express, "
                    "'backend'→node, 'py'→python, 'rn'→react-native."
                ),
            )

        step_results, all_ok = _scaffold_dir(detected_type, safe_name, abs_out, timeout,
                                             progress_callback=getattr(self, "_progress_callback", None))

        if all_ok:
            _patch_package_json(abs_out, safe_name)

        initialized = all_ok and os.path.isdir(abs_out)
        rename_note = (
            f"  (directory renamed: '{dir_basename}' → '{safe_dir_name}')"
            if renamed_dir else ""
        )
        return ToolResult(
            tool_name=self.name,
            success=all_ok,
            output={
                "architecture": arch, "project_type": detected_type,
                "project_name": safe_name, "project_path": abs_out,
                "output_dir": abs_out, "initialized": initialized,
                "renamed_dir": renamed_dir, "steps": step_results,
                "message": (
                    f"Project '{safe_name}' ({detected_type}) initialised in {abs_out}.{rename_note}"
                    if all_ok else
                    f"Initialisation failed at: {step_results[-1]['command']}"
                ),
            },
            error=None if all_ok else step_results[-1].get("stderr", "Unknown error"),
            metadata={
                "architecture": arch, "detected_type": detected_type,
                "safe_name": safe_name, "renamed_dir": renamed_dir,
            },
        )

    # ------------------------------------------------------------------
    # Fullstack scaffolder
    # ------------------------------------------------------------------

    def _run_fullstack(
        self,
        safe_name: str,
        frontend_type: str,
        backend_type: str,
        abs_out: str,
        dir_basename: str,
        renamed_dir: Optional[str],
        timeout: int,
        progress_callback: Optional[Any] = None,
    ) -> ToolResult:
        """Scaffold frontend/ and backend/ subdirectories inside *abs_out*."""
        # Guard: fall back to defaults if types are not known
        frontend_type = _TYPE_ALIASES.get(frontend_type, frontend_type) or "react"
        backend_type  = _TYPE_ALIASES.get(backend_type,  backend_type)  or "express"
        if frontend_type not in _SCAFFOLD_RECIPES:
            frontend_type = "react"
        if backend_type not in _SCAFFOLD_RECIPES:
            backend_type = "express"

        fe_dir = os.path.join(abs_out, "frontend")
        be_dir = os.path.join(abs_out, "backend")
        os.makedirs(fe_dir, exist_ok=True)
        os.makedirs(be_dir, exist_ok=True)

        fe_steps, fe_ok = _scaffold_dir(frontend_type, safe_name, fe_dir, timeout,
                                          progress_callback=progress_callback)
        be_steps, be_ok = _scaffold_dir(backend_type,  safe_name, be_dir, timeout,
                                          progress_callback=progress_callback)

        if fe_ok:
            _patch_package_json(fe_dir, safe_name)
        if be_ok:
            _patch_package_json(be_dir, safe_name)

        all_ok = fe_ok and be_ok
        rename_note = (
            f"  (parent directory renamed: '{dir_basename}' → '{os.path.basename(abs_out)}')"
            if renamed_dir else ""
        )

        return ToolResult(
            tool_name=self.name,
            success=all_ok,
            output={
                "architecture":  "fullstack",
                "project_name":  safe_name,
                "project_path":  abs_out,
                "output_dir":    abs_out,
                "frontend_dir":  fe_dir,
                "backend_dir":   be_dir,
                "frontend_type": frontend_type,
                "backend_type":  backend_type,
                "initialized":   all_ok,
                "renamed_dir":   renamed_dir,
                "steps":         {"frontend": fe_steps, "backend": be_steps},
                "message": (
                    f"Fullstack '{safe_name}' initialised.{rename_note}\n"
                    f"  frontend ({frontend_type}): {fe_dir}\n"
                    f"  backend  ({backend_type}):  {be_dir}"
                    if all_ok else
                    "Fullstack initialisation partially failed — see steps for details."
                ),
            },
            error=None if all_ok else "One or more scaffold steps failed.",
            metadata={
                "architecture":  "fullstack",
                "frontend_type": frontend_type,
                "backend_type":  backend_type,
                "safe_name":     safe_name,
                "renamed_dir":   renamed_dir,
            },
        )# Changed tools/project_initializer.py
