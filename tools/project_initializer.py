"""project_initializer.py — Sentinel ProjectInitializer tool.

Detects the project type from a goal/description and runs the appropriate
scaffolding command (e.g. ``npx create-react-app``, ``npm init``,
``flutter create``, ``gradle init``, etc.) before any source files are
written.

Supported project types
-----------------------
Web / Frontend
  react          — npx create-react-app <name>
  react-ts       — npx create-react-app <name> --template typescript
  vite           — npm create vite@latest <name>
  nextjs         — npx create-next-app@latest <name>
  vue            — npm create vue@latest <name>
  angular        — npx @angular/cli new <name>
  svelte         — npm create svelte@latest <name>

Backend / Node
  node           — npm init -y
  express        — npm init -y + npm install express
  fastify        — npm init -y + npm install fastify

Backend / Python
  python         — python -m venv venv + pip init stub
  fastapi        — pip install fastapi uvicorn (venv)
  django         — django-admin startproject <name>
  flask          — pip install flask (venv)

Mobile
  react-native   — npx react-native@latest init <name>
  expo           — npx create-expo-app <name>
  flutter        — flutter create <name>
  kotlin-android — gradle init (Android project stub)
  swift-ios      — xcodebuild / swift package init

Desktop / Game
  unity          — Warns user to use Unity Hub CLI
  unreal         — Warns user to use Unreal Engine CLI
  godot          — Warns user to use Godot editor
  tauri          — npm create tauri-app@latest <name>
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
    "react": [
        "npx --yes create-react-app {name}",
    ],
    "react-ts": [
        "npx --yes create-react-app {name} --template typescript",
    ],
    "vite": [
        "npm create --yes vite@latest {name} -- --template react",
        "cd {name} && npm install",
    ],
    "nextjs": [
        "npx --yes create-next-app@latest {name} --no-interactive",
    ],
    "vue": [
        "npm create --yes vue@latest {name}",
        "cd {name} && npm install",
    ],
    "angular": [
        "npx --yes @angular/cli new {name} --skip-git --skip-install",
        "cd {name} && npm install",
    ],
    "svelte": [
        "npm create --yes svelte@latest {name}",
        "cd {name} && npm install",
    ],
    # ── Backend / Node ───────────────────────────────────────────────────────
    "node": [
        "mkdir -p {name}",
        "cd {name} && npm init -y",
    ],
    "express": [
        "mkdir -p {name}",
        "cd {name} && npm init -y",
        "cd {name} && npm install express",
    ],
    "fastify": [
        "mkdir -p {name}",
        "cd {name} && npm init -y",
        "cd {name} && npm install fastify",
    ],
    # ── Backend / Python ─────────────────────────────────────────────────────
    "python": [
        "mkdir -p {name}",
        "python -m venv {name}/venv",
    ],
    "fastapi": [
        "mkdir -p {name}",
        "python -m venv {name}/venv",
        "{pip} install fastapi uvicorn[standard]",
    ],
    "django": [
        "python -m venv {name}_env",
        "{pip} install django",
        "{django_admin} startproject {name}",
    ],
    "flask": [
        "mkdir -p {name}",
        "python -m venv {name}/venv",
        "{pip} install flask",
    ],
    # ── Mobile ───────────────────────────────────────────────────────────────
    "react-native": [
        "npx --yes react-native@latest init {name}",
    ],
    "expo": [
        "npx --yes create-expo-app@latest {name}",
    ],
    "flutter": [
        "flutter create {name}",
    ],
    "kotlin-android": [
        "mkdir -p {name}",
        "cd {name} && gradle init --type basic --dsl kotlin --project-name {name} --no-incubating",
    ],
    "swift-ios": [
        "mkdir -p {name}",
        "cd {name} && swift package init --name {name} --type executable",
    ],
    # ── Desktop ──────────────────────────────────────────────────────────────
    "electron": [
        "mkdir -p {name}",
        "cd {name} && npm init -y",
        "cd {name} && npm install --save-dev electron",
    ],
    "tauri": [
        "npm create --yes tauri-app@latest {name}",
    ],
    # ── Game Engines (require GUI / proprietary CLI — emit guidance only) ────
    "unity":  [],   # handled separately with a user-facing message
    "unreal": [],
    "godot":  [],
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


def _pip_executable(project_name: str, cwd: str) -> str:
    """Return the pip path inside the project's venv if it exists."""
    venv_dir = os.path.join(cwd, project_name, "venv")
    if sys.platform == "win32":
        candidate = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        candidate = os.path.join(venv_dir, "bin", "pip")
    return candidate if os.path.isfile(candidate) else "pip"


def _django_admin_executable(project_name: str, cwd: str) -> str:
    venv_dir = os.path.join(cwd, f"{project_name}_env")
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
) -> Tuple[bool, str, str]:
    """Run a shell command and return (success, stdout, stderr)."""
    _env = dict(os.environ)
    # Suppress interactive prompts
    _env["CI"] = "true"
    _env["DEBIAN_FRONTEND"] = "noninteractive"
    if env:
        _env.update(env)

    try:
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
        "(e.g. npx create-react-app, flutter create, npm init, django-admin startproject). "
        "Call this BEFORE writing any source files for a brand-new project."
    )
    parameters_schema = {
        "project_name": {
            "type": "string",
            "description": "Name / slug for the new project folder.",
            "required": True,
        },
        "project_type": {
            "type": "string",
            "description": (
                "Explicit project type. One of: react, react-ts, vite, nextjs, vue, "
                "angular, svelte, node, express, fastify, python, fastapi, django, flask, "
                "react-native, expo, flutter, kotlin-android, swift-ios, "
                "electron, tauri, unity, unreal, godot. "
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
            "description": "Parent directory in which to create the project. Defaults to '.'.",
            "required": False,
            "default": ".",
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
        timeout: int = _LONG_TIMEOUT,
        **_: Any,
    ) -> ToolResult:
        # Sanitise project name — alphanumeric + hyphens / underscores only
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "-", project_name).strip("-") or "my-project"

        # Resolve / create output directory
        abs_out = os.path.realpath(output_dir)
        os.makedirs(abs_out, exist_ok=True)

        # Detect project type
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

        # --- GUI-only game engines ---
        if detected_type in _ENGINE_GUIDANCE:
            guidance = _ENGINE_GUIDANCE[detected_type].format(name=safe_name)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={
                    "project_type": detected_type,
                    "project_name": safe_name,
                    "output_dir": abs_out,
                    "initialized": False,
                    "guidance": guidance,
                    "steps": [],
                    "message": (
                        f"{detected_type.capitalize()} projects cannot be initialised from "
                        "the command line without the engine installed. "
                        "See 'guidance' for manual steps."
                    ),
                },
            )

        if detected_type not in _SCAFFOLD_RECIPES:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Unknown project type '{detected_type}'. "
                    f"Supported: {', '.join(sorted(_SCAFFOLD_RECIPES))}."
                ),
            )

        commands = _SCAFFOLD_RECIPES[detected_type]
        if not commands:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"No scaffold recipe defined for project type '{detected_type}'.",
            )

        # Resolve pip / django-admin paths (may not exist yet pre-venv creation)
        pip = _pip_executable(safe_name, abs_out)
        django_admin = _django_admin_executable(safe_name, abs_out)

        step_results: List[Dict[str, Any]] = []
        all_ok = True

        for raw_cmd in commands:
            cmd = raw_cmd.format(
                name=safe_name,
                path=abs_out,
                pip=pip,
                django_admin=django_admin,
            )
            # Re-resolve pip after venv may have been created by a previous step
            pip = _pip_executable(safe_name, abs_out)
            django_admin = _django_admin_executable(safe_name, abs_out)

            ok, stdout, stderr = _run_command(cmd, cwd=abs_out, timeout=timeout)
            step_results.append({
                "command": cmd,
                "success": ok,
                "stdout": stdout[:4000],
                "stderr": stderr[:2000],
            })
            if not ok:
                all_ok = False
                break  # stop on first failure

        project_path = os.path.join(abs_out, safe_name)
        initialized = all_ok and os.path.isdir(project_path)

        return ToolResult(
            tool_name=self.name,
            success=all_ok,
            output={
                "project_type": detected_type,
                "project_name": safe_name,
                "project_path": project_path,
                "output_dir": abs_out,
                "initialized": initialized,
                "steps": step_results,
                "message": (
                    f"Project '{safe_name}' ({detected_type}) initialised successfully at {project_path}."
                    if all_ok else
                    f"Initialisation failed at step: {step_results[-1]['command']}"
                ),
            },
            error=None if all_ok else step_results[-1].get("stderr", "Unknown error"),
            metadata={"detected_type": detected_type, "safe_name": safe_name},
        )
