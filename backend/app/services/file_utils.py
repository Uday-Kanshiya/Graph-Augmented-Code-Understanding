from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from app.models.schemas import RepoFile

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".next",
}


def clean_repo_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return cleaned or "repo"


def read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def source_snippet(text: str, line_start: int, line_end: int, max_chars: int | None = 800) -> str:
    lines = text.splitlines()
    start = max(0, line_start - 1)
    end = min(len(lines), line_end)
    snippet = "\n".join(lines[start:end])
    if max_chars is not None and len(snippet) > max_chars:
        return snippet[:max_chars] + "\n..."
    return snippet


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

BINARY_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".webp", ".psd", ".ai",
    # Audio/Video
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm",
    # Archives / Compressed
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
    # Executables / Binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".out", ".app", ".sys",
    # Databases
    ".db", ".sqlite", ".sqlite3", ".sqlitedb",
    # Office documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Misc/Java/Python
    ".class", ".jar", ".war", ".ear", ".pyc", ".pyo", ".pyd",
}


def is_binary_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
    except Exception:
        return True
    return False


def iter_code_files(root: Path) -> list[RepoFile]:
    files: list[RepoFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if is_ignored(rel):
            continue
        if is_binary_file(path):
            continue
        
        suffix = path.suffix.lower()
        if suffix in EXTENSION_TO_LANGUAGE:
            language = EXTENSION_TO_LANGUAGE[suffix]
        else:
            # Map dynamic suffix or default to text
            language = suffix.lstrip(".") if suffix else "text"
            if not language.isalnum():
                language = "text"
                
        try:
            text = read_text_lossy(path)
            files.append(
                RepoFile(
                    path=rel.as_posix(),
                    language=language,
                    size_bytes=path.stat().st_size,
                    line_count=len(text.splitlines()),
                )
            )
        except Exception:
            continue
    return files


def iter_python_files(root: Path) -> list[RepoFile]:
    return iter_code_files(root)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = destination / member.filename
            resolved = member_path.resolve()
            if destination_resolved not in resolved.parents and resolved != destination_resolved:
                raise ValueError(f"Unsafe zip path detected: {member.filename}")
            if is_ignored(Path(member.filename)):
                continue
            archive.extract(member, destination)
    _flatten_single_top_level_dir(destination)


def _flatten_single_top_level_dir(destination: Path) -> None:
    children = [child for child in destination.iterdir() if child.name not in {"__MACOSX"}]
    if len(children) != 1 or not children[0].is_dir():
        return
    inner = children[0]
    tmp = destination.with_name(destination.name + "-flattening")
    if tmp.exists():
        shutil.rmtree(tmp)
    inner.rename(tmp)
    shutil.rmtree(destination)
    tmp.rename(destination)


def validate_github_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("GitHub URL must use http or https.")
    if parsed.netloc.lower() != "github.com":
        raise ValueError("Stage 1 Git import only accepts github.com URLs.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository.")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not owner or not repo:
        raise ValueError("GitHub URL must include owner and repository.")
    return f"https://github.com/{owner}/{repo}.git"

