"""Project persistence and subtitle/lyrics export helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"

_LOCK = threading.RLock()


def ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def new_project_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def project_dir(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", project_id):
        raise ValueError(f"Invalid project id: {project_id}")
    return PROJECTS_DIR / project_id


def _meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "meta.json"


def save_project(meta: dict[str, Any]) -> None:
    ensure_dirs()
    pid = meta["id"]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    path = _meta_path(pid)
    tmp_path = pdir / f"meta.json.{threading.get_ident()}.tmp"
    with _LOCK:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)


def load_project(project_id: str) -> Optional[dict[str, Any]]:
    path = _meta_path(project_id)
    if not path.exists():
        return None
    with _LOCK:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def update_project(project_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    with _LOCK:
        meta = load_project(project_id)
        if meta is None:
            return None
        meta.update(fields)
        save_project(meta)
        return meta


def list_projects() -> list[dict[str, Any]]:
    ensure_dirs()
    items: list[dict[str, Any]] = []
    for entry in PROJECTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append(
                {
                    "id": meta["id"],
                    "title": meta.get("title", "未命名听力"),
                    "created_at": meta.get("created_at", 0),
                    "duration": meta.get("duration", 0.0),
                    "sentence_count": len(meta.get("sentences", [])),
                    "status": meta.get("status", "ready"),
                    "audio_filename": meta.get("audio_filename", ""),
                    "transcript_filename": meta.get("transcript_filename", ""),
                    "language": meta.get("language", "auto"),
                    "model_size": meta.get("model_size", "base"),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return items


def delete_project(project_id: str) -> bool:
    pdir = project_dir(project_id)
    if not pdir.exists():
        return False
    with _LOCK:
        shutil.rmtree(pdir, ignore_errors=True)
    return True


def _format_lrc_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:05.2f}"


def _format_srt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    hours = int(sec // 3600)
    mins = int((sec % 3600) // 60)
    secs = int(sec % 60)
    millis = int(round((sec - int(sec)) * 1000))
    if millis >= 1000:
        millis -= 1000
        secs += 1
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def export_lrc(meta: dict[str, Any]) -> str:
    lines = [
        f"[ti:{meta.get('title', 'Listening')}]",
        "[by:ListenSync]",
        "",
    ]
    for s in meta.get("sentences", []):
        if s.get("kind") == "header":
            continue
        speaker = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"[{_format_lrc_time(s['start'])}]{speaker}{s['text']}")
    return "\n".join(lines) + "\n"


def export_srt(meta: dict[str, Any]) -> str:
    blocks: list[str] = []
    idx = 1
    for s in meta.get("sentences", []):
        if s.get("kind") == "header":
            continue
        speaker = f"{s['speaker']}: " if s.get("speaker") else ""
        start = _format_srt_time(s["start"])
        end = _format_srt_time(max(s["start"] + 0.1, s["end"]))
        blocks.append(f"{idx}\n{start} --> {end}\n{speaker}{s['text']}")
        idx += 1
    return "\n\n".join(blocks) + "\n"
