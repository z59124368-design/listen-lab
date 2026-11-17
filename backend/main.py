"""FastAPI application for ListenSync MP3+PDF synced listening player."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .aligner import align_audio_and_transcript
from .pdf_parser import parse_pdf_transcript, parse_plain_transcript
from .storage import (
    delete_project,
    ensure_dirs,
    export_lrc,
    export_srt,
    list_projects,
    load_project,
    new_project_id,
    project_dir,
    save_project,
    update_project,
)


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".webm", ".mp4"}

app = FastAPI(title="ListenSync API", version="1.0.0")

# Track running alignment threads to avoid duplicate concurrent jobs per project
_ACTIVE_JOBS: set[str] = set()
_JOBS_LOCK = threading.Lock()


class SentenceUpdate(BaseModel):
    text: Optional[str] = None
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


class SentenceSplitRequest(BaseModel):
    split_index: int
    split_time: Optional[float] = None


class RealignRequest(BaseModel):
    model_size: str = "base"
    language: str = "auto"


def _audio_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }
    if ext in mapping:
        return mapping[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _run_alignment_job(project_id: str, audio_path: str, model_size: str, language: str) -> None:
    with _JOBS_LOCK:
        if project_id in _ACTIVE_JOBS:
            return
        _ACTIVE_JOBS.add(project_id)

    def on_progress(pct: int, msg: str) -> None:
        update_project(project_id, progress=pct, progress_message=msg)

    try:
        meta = load_project(project_id)
        if meta is None:
            return
        pdir = str(project_dir(project_id))
        items = meta.get("sentences", [])
        # Strip prior timestamps before re-aligning
        clean_items = []
        for it in items:
            copy_it = dict(it)
            copy_it["start"] = 0.0
            copy_it["end"] = 0.0
            copy_it["confidence"] = 0.0 if copy_it.get("kind") != "header" else 1.0
            clean_items.append(copy_it)

        result = align_audio_and_transcript(
            audio_path=audio_path,
            work_dir=pdir,
            items=clean_items,
            model_size=model_size,
            language=language,
            on_progress=on_progress,
        )
        update_project(
            project_id,
            status="ready",
            progress=100,
            progress_message="对齐完成",
            error=None,
            duration=result["duration"],
            detected_language=result["detected_language"],
            waveform=result["waveform"],
            sentences=result["sentences"],
            model_size=model_size,
            language=language,
        )
    except Exception as exc:
        update_project(
            project_id,
            status="error",
            progress_message="对齐失败",
            error=str(exc),
        )
    finally:
        with _JOBS_LOCK:
            _ACTIVE_JOBS.discard(project_id)


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/projects")
def api_list_projects() -> list[dict[str, Any]]:
    return list_projects()


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str) -> dict[str, Any]:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到该听力项目")
    return meta


@app.post("/api/projects")
async def api_create_project(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    pdf: Optional[UploadFile] = File(None),
    transcript_text: str = Form(""),
    title: str = Form(""),
    language: str = Form("auto"),
    model_size: str = Form("base"),
) -> dict[str, Any]:
    audio_name = audio.filename or "audio.mp3"
    audio_ext = Path(audio_name).suffix.lower() or ".mp3"
    if audio_ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {audio_ext}")

    has_pdf = pdf is not None and bool(pdf.filename)
    has_text = bool(transcript_text and transcript_text.strip())
    if not has_pdf and not has_text:
        raise HTTPException(status_code=400, detail="请上传 PDF 原文文件，或输入听力原文文本")

    pid = new_project_id()
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path = pdir / f"audio{audio_ext}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        transcript_filename = "pasted_text.txt"
        items: list[dict[str, Any]] = []

        if has_pdf and pdf is not None:
            transcript_filename = pdf.filename or "transcript.pdf"
            t_ext = Path(transcript_filename).suffix.lower()
            if t_ext == ".txt":
                raw_bytes = await pdf.read()
                txt = raw_bytes.decode("utf-8", errors="replace")
                (pdir / "transcript.txt").write_text(txt, encoding="utf-8")
                items = parse_plain_transcript(txt)
            elif t_ext == ".pdf":
                pdf_path = pdir / "transcript.pdf"
                with open(pdf_path, "wb") as f:
                    shutil.copyfileobj(pdf.file, f)
                items = parse_pdf_transcript(str(pdf_path))
            else:
                raise HTTPException(status_code=400, detail="原文文件请上传 .pdf 或 .txt 格式")
        else:
            (pdir / "transcript.txt").write_text(transcript_text, encoding="utf-8")
            items = parse_plain_transcript(transcript_text)

        spoken_count = sum(1 for it in items if it.get("kind") != "header")
        if spoken_count == 0:
            raise HTTPException(
                status_code=400,
                detail="未能从原文中提取到句子。如果是扫描版图片 PDF，请先 OCR 转为文字版，或直接粘贴原文文本。",
            )

        clean_title = title.strip() or Path(audio_name).stem
        meta: dict[str, Any] = {
            "id": pid,
            "title": clean_title,
            "created_at": time.time(),
            "audio_filename": audio_name,
            "audio_relpath": audio_path.name,
            "transcript_filename": transcript_filename,
            "language": language,
            "model_size": model_size,
            "detected_language": language,
            "duration": 0.0,
            "status": "processing",
            "progress": 5,
            "progress_message": f"已提取 {spoken_count} 个句子，准备开始时间轴对齐...",
            "error": None,
            "waveform": [],
            "sentences": items,
        }
        save_project(meta)
        background_tasks.add_task(_run_alignment_job, pid, str(audio_path), model_size, language)
        return {"id": pid, "status": "processing", "sentence_count": spoken_count}
    except HTTPException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"创建项目失败: {exc}") from exc


@app.get("/api/projects/{project_id}/audio")
def api_stream_audio(project_id: str) -> FileResponse:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到该听力项目")
    audio_rel = meta.get("audio_relpath", "audio.mp3")
    audio_path = project_dir(project_id) / audio_rel
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(
        path=audio_path,
        media_type=_audio_media_type(audio_path),
        filename=meta.get("audio_filename", audio_path.name),
    )


@app.post("/api/projects/{project_id}/realign")
def api_realign_project(
    project_id: str,
    req: RealignRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到该听力项目")
    with _JOBS_LOCK:
        if project_id in _ACTIVE_JOBS:
            raise HTTPException(status_code=409, detail="该项目正在对齐中，请稍候")

    audio_path = project_dir(project_id) / meta.get("audio_relpath", "audio.mp3")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")

    update_project(
        project_id,
        status="processing",
        progress=5,
        progress_message="正在准备重新对齐...",
        error=None,
    )
    background_tasks.add_task(_run_alignment_job, project_id, str(audio_path), req.model_size, req.language)
    return {"id": project_id, "status": "processing"}


@app.patch("/api/projects/{project_id}/sentences/{sentence_id}")
def api_update_sentence(project_id: str, sentence_id: int, body: SentenceUpdate) -> dict[str, Any]:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到项目")
    sentences = meta.get("sentences", [])
    target = next((s for s in sentences if s["id"] == sentence_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="未找到对应句子")

    if body.text is not None:
        target["text"] = body.text.strip()
    if body.speaker is not None:
        target["speaker"] = body.speaker.strip()
    duration = float(meta.get("duration") or 0.0)
    if body.start is not None:
        target["start"] = round(max(0.0, min(duration or 99999.0, body.start)), 3)
    if body.end is not None:
        target["end"] = round(max(target["start"] + 0.1, min(duration or 99999.0, body.end)), 3)
    if target["end"] <= target["start"]:
        target["end"] = round(target["start"] + 0.25, 3)

    save_project(meta)
    return target


@app.post("/api/projects/{project_id}/sentences/{sentence_id}/split")
def api_split_sentence(project_id: str, sentence_id: int, body: SentenceSplitRequest) -> dict[str, Any]:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到项目")
    sentences = meta.get("sentences", [])
    idx = next((i for i, s in enumerate(sentences) if s["id"] == sentence_id), None)
    if idx is None or sentences[idx].get("kind") == "header":
        raise HTTPException(status_code=404, detail="未找到可拆分的句子")

    cur = sentences[idx]
    text = cur["text"]
    pos = max(1, min(len(text) - 1, body.split_index))
    left_txt = text[:pos].strip()
    right_txt = text[pos:].strip()
    if not left_txt or not right_txt:
        raise HTTPException(status_code=400, detail="拆分位置两侧均不能为空")

    s_start = float(cur["start"])
    s_end = float(cur["end"])
    if body.split_time is not None and s_start + 0.1 < body.split_time < s_end - 0.1:
        mid_t = round(float(body.split_time), 3)
    else:
        ratio = len(left_txt) / max(1, len(left_txt) + len(right_txt))
        mid_t = round(s_start + (s_end - s_start) * ratio, 3)

    first = dict(cur)
    first["text"] = left_txt
    first["end"] = mid_t

    second = dict(cur)
    second["text"] = right_txt
    second["start"] = mid_t
    second["end"] = max(mid_t + 0.15, s_end)

    new_list = sentences[:idx] + [first, second] + sentences[idx + 1 :]
    for new_id, item in enumerate(new_list):
        item["id"] = new_id
    meta["sentences"] = new_list
    save_project(meta)
    return {"sentences": new_list}


@app.post("/api/projects/{project_id}/sentences/{sentence_id}/merge-next")
def api_merge_next_sentence(project_id: str, sentence_id: int) -> dict[str, Any]:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到项目")
    sentences = meta.get("sentences", [])
    idx = next((i for i, s in enumerate(sentences) if s["id"] == sentence_id), None)
    if idx is None or sentences[idx].get("kind") == "header":
        raise HTTPException(status_code=404, detail="未找到对应句子")

    next_idx = next((i for i in range(idx + 1, len(sentences)) if sentences[i].get("kind") != "header"), None)
    if next_idx is None:
        raise HTTPException(status_code=400, detail="后面没有可合并的句子")

    cur = sentences[idx]
    nxt = sentences[next_idx]
    cur["text"] = f"{cur['text']} {nxt['text']}".strip()
    cur["end"] = max(float(cur["end"]), float(nxt["end"]))
    cur["confidence"] = round(0.5 * (float(cur.get("confidence", 0.5)) + float(nxt.get("confidence", 0.5))), 2)

    new_list = sentences[:next_idx] + sentences[next_idx + 1 :]
    for new_id, item in enumerate(new_list):
        item["id"] = new_id
    meta["sentences"] = new_list
    save_project(meta)
    return {"sentences": new_list}


def _content_disposition(filename: str) -> str:
    ascii_name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", filename).strip() or "listening"
    encoded = quote(filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


@app.get("/api/projects/{project_id}/export")
def api_export(project_id: str, format: str = Query("lrc")) -> PlainTextResponse:
    meta = load_project(project_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="未找到项目")
    title = meta.get("title", "listening").replace('"', "").strip() or "listening"
    fmt = format.lower()
    if fmt == "lrc":
        content = export_lrc(meta)
        return PlainTextResponse(
            content,
            headers={"Content-Disposition": _content_disposition(f"{title}.lrc")},
        )
    if fmt == "srt":
        content = export_srt(meta)
        return PlainTextResponse(
            content,
            headers={"Content-Disposition": _content_disposition(f"{title}.srt")},
        )
    if fmt == "json":
        content = json.dumps(meta.get("sentences", []), ensure_ascii=False, indent=2)
        return PlainTextResponse(
            content,
            media_type="application/json",
            headers={"Content-Disposition": _content_disposition(f"{title}.json")},
        )
    raise HTTPException(status_code=400, detail="不支持的导出格式，请选择 lrc / srt / json")


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str) -> dict[str, bool]:
    ok = delete_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到项目")
    return {"deleted": True}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
