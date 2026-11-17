"""Comprehensive end-to-end tests for ListenSync PDF parser, forced aligner, and HTTP API."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import av
import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.aligner import align_audio_and_transcript, align_tokens, TimedToken, normalize_text_to_tokens
from backend.pdf_parser import parse_pdf_transcript, parse_plain_transcript, split_sentences


def test_sentence_splitting_and_abbreviations() -> None:
    text = (
        "Dr. Smith arrived at 9:30 a.m. in the U.S. yesterday. "
        "He paid £12.50 for a book! Wait... did he buy two?"
    )
    sents = split_sentences(text)
    assert len(sents) == 3, f"Expected 3 sentences, got {sents}"
    assert sents[0].startswith("Dr. Smith")
    assert sents[1].startswith("He paid")
    assert sents[2].startswith("Wait...")


def test_pdf_parser_headers_columns_and_speakers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "sample.pdf"
        doc = pymupdf.open()

        # Page 1: repeated header/footer, page number, section, hyphen break, speaker turns
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 35), "Cambridge Listening Practice - Audioscript", fontsize=10)
        p1.insert_text((50, 110), "PART 1", fontsize=13)
        p1.insert_text(
            (50, 145),
            "RECEPTIONIST: Good morning, Westgate Sports Cen-\ntre. How can I help you today?",
            fontsize=11,
        )
        p1.insert_text(
            (50, 210),
            "CALLER: Hi, I'd like to book a badminton court for Sat-\nurday morning at 10:30.",
            fontsize=11,
        )
        p1.insert_text((285, 810), "Page 1 of 2", fontsize=10)

        # Page 2: repeated header/footer + 2-column layout
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 35), "Cambridge Listening Practice - Audioscript", fontsize=10)
        p2.insert_text((50, 110), "RECEPTIONIST: Certainly, let me check.", fontsize=11)
        p2.insert_text((50, 145), "The fee is $25 per hour.", fontsize=11)
        p2.insert_text((320, 110), "CALLER: That sounds fine.", fontsize=11)
        p2.insert_text((320, 145), "My name is Mr. David Johnson.", fontsize=11)
        p2.insert_text((285, 810), "Page 2 of 2", fontsize=10)

        doc.save(str(pdf_path))
        doc.close()

        items = parse_pdf_transcript(str(pdf_path))
        spoken = [x for x in items if x["kind"] != "header"]
        headers = [x for x in items if x["kind"] == "header"]

        assert len(headers) == 1 and headers[0]["text"] == "PART 1", headers
        assert len(spoken) == 7, f"Expected 7 spoken sentences, got {spoken}"
        assert "Centre" in spoken[0]["text"], spoken[0]
        assert spoken[0]["speaker"] == "RECEPTIONIST"
        assert "Saturday" in spoken[2]["text"], spoken[2]
        assert spoken[2]["speaker"] == "CALLER"
        assert spoken[3]["speaker"] == "RECEPTIONIST"
        assert spoken[6]["speaker"] == "CALLER"
        assert "Mr. David Johnson." in spoken[6]["text"]


def test_normalization_and_dp_alignment() -> None:
    toks_pdf = normalize_text_to_tokens("I don't think 1998 cost $25 on 1st May.")
    toks_asr = normalize_text_to_tokens("I do not think nineteen ninety eight cost twenty five dollars on first May")
    assert toks_pdf == toks_asr, (toks_pdf, toks_asr)

    cjk_toks = normalize_text_to_tokens("今天天气很好。")
    assert cjk_toks == ["今", "天", "天", "气", "很", "好"]

    # Test semi-global DP skips spoken intro not present in PDF
    asr_words = [
        TimedToken("test", 0.5, 0.9),
        TimedToken("one", 0.9, 1.3),
        TimedToken("you", 1.5, 1.7),
        TimedToken("will", 1.7, 1.9),
        TimedToken("hear", 1.9, 2.2),
        TimedToken("good", 3.5, 3.8),
        TimedToken("morning", 3.8, 4.2),
        TimedToken("everyone", 4.2, 4.7),
    ]
    p_toks = ["good", "morning", "everyone"]
    mapping = align_tokens(p_toks, asr_words)
    assert mapping == [5, 6, 7], f"Unexpected mapping: {mapping}"


def _synthesize_spoken_mp3(segments: list[str], out_mp3: Path) -> list[tuple[float, float]]:
    """Synthesize spoken audio with macOS `say` and concatenate with silence gaps into an MP3."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pcm_parts: list[np.ndarray] = []
        gt_spans: list[tuple[float, float]] = []
        sr = 16000
        cur_time = 0.0

        # Leading silence 0.5s
        lead = np.zeros(int(0.5 * sr), dtype=np.int16)
        pcm_parts.append(lead)
        cur_time += 0.5

        for idx, text in enumerate(segments):
            aiff = tmp_dir / "seg.aiff"
            wav = tmp_dir / "seg.wav"
            voice = "Daniel" if idx % 2 == 0 else "Fred"
            subprocess.run(["say", "-v", voice, "-r", "165", "-o", str(aiff), text], check=True)
            subprocess.run(
                ["/usr/bin/afconvert", str(aiff), "-d", "LEI16@16000", "-c", "1", "-f", "WAVE", str(wav)],
                check=True,
            )
            with wave.open(str(wav), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16)
            # Trim dead digital zeros at start/end of TTS clip to know true speech bounds
            nz = np.where(np.abs(arr) > 120)[0]
            if len(nz) > 0:
                arr = arr[max(0, nz[0] - 800) : min(len(arr), nz[-1] + 800)]

            start_t = cur_time
            dur_t = len(arr) / sr
            end_t = start_t + dur_t
            gt_spans.append((start_t, end_t))
            pcm_parts.append(arr)
            cur_time = end_t

            # 0.75s silence gap between sentences
            gap = np.zeros(int(0.75 * sr), dtype=np.int16)
            pcm_parts.append(gap)
            cur_time += 0.75

        full_pcm = np.concatenate(pcm_parts)
        with av.open(str(out_mp3), mode="w") as out:
            stream = out.add_stream("libmp3lame", rate=sr)
            stream.layout = "mono"
            frame_size = 1152
            for idx in range(0, len(full_pcm), frame_size):
                chunk = full_pcm[idx : idx + frame_size]
                frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
                frame.sample_rate = sr
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)

        return gt_spans


def _encode_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----ListenSyncBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    for name, (fname, data, ctype) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'.encode())
        body.extend(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_end_to_end_server_and_alignment() -> None:
    intro = "Cambridge practice test one. Part one."
    spoken_lines = [
        "Good morning, Westgate Sports Centre.",
        "Hello, I would like to book a tennis court for Saturday.",
        "Certainly, what time would you prefer?",
        "Around ten thirty in the morning if possible.",
        "That costs twenty five dollars per hour.",
        "Great, please reserve it under the name David Brown.",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mp3_path = tmp_dir / "listening_demo.mp3"
        pdf_path = tmp_dir / "transcript_demo.pdf"

        gt_spans = _synthesize_spoken_mp3([intro] + spoken_lines, mp3_path)
        sentence_gt = gt_spans[1:]  # skip the spoken intro not present in PDF

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 40), "Listening Transcript Demo", fontsize=10)
        page.insert_text((50, 95), "PART 1", fontsize=13)
        page.insert_text((50, 130), f"RECEPTIONIST: {spoken_lines[0]}", fontsize=11)
        page.insert_text((50, 165), f"CALLER: {spoken_lines[1]}", fontsize=11)
        page.insert_text((50, 200), f"RECEPTIONIST: {spoken_lines[2]}", fontsize=11)
        page.insert_text((50, 235), f"CALLER: {spoken_lines[3]}", fontsize=11)
        page.insert_text((50, 270), f"RECEPTIONIST: {spoken_lines[4]}", fontsize=11)
        page.insert_text((50, 305), f"CALLER: {spoken_lines[5]}", fontsize=11)
        page.insert_text((280, 800), "1", fontsize=10)
        doc.save(str(pdf_path))
        doc.close()

        port = _free_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        project_id = None
        try:
            base_url = f"http://127.0.0.1:{port}"
            # Wait for server ready
            for _ in range(40):
                try:
                    with urllib.request.urlopen(base_url + "/api/projects", timeout=1) as resp:
                        if resp.status == 200:
                            break
                except Exception:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Server failed to start")

            # Upload MP3 + PDF
            body, ctype = _encode_multipart(
                fields={"title": "测试听力 Test 1", "language": "en", "model_size": "base"},
                files={
                    "audio": ("listening_demo.mp3", mp3_path.read_bytes(), "audio/mpeg"),
                    "pdf": ("transcript_demo.pdf", pdf_path.read_bytes(), "application/pdf"),
                },
            )
            req = urllib.request.Request(
                base_url + "/api/projects",
                data=body,
                headers={"Content-Type": ctype},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                created = json.loads(resp.read().decode("utf-8"))
            project_id = created["id"]
            assert created["sentence_count"] == 6, created

            # Poll until ready
            meta = None
            for _ in range(90):
                with urllib.request.urlopen(base_url + f"/api/projects/{project_id}", timeout=5) as resp:
                    meta = json.loads(resp.read().decode("utf-8"))
                if meta["status"] in ("ready", "error"):
                    break
                time.sleep(0.5)

            assert meta is not None and meta["status"] == "ready", meta
            spoken = [s for s in meta["sentences"] if s["kind"] != "header"]
            assert len(spoken) == 6, spoken
            assert len(meta["waveform"]) >= 100

            # Verify alignment accuracy against ground truth!
            for idx, (sent, (gt_s, gt_e)) in enumerate(zip(spoken, sentence_gt)):
                start_err = abs(sent["start"] - gt_s)
                end_err = abs(sent["end"] - gt_e)
                print(
                    f"  Sentence {idx+1}: aligned=[{sent['start']:.2f}, {sent['end']:.2f}] "
                    f"gt=[{gt_s:.2f}, {gt_e:.2f}] err=({start_err:.2f}s, {end_err:.2f}s) conf={sent['confidence']}"
                )
                assert start_err <= 0.45, f"Start error too high for sentence {idx}: {start_err}"
                assert end_err <= 0.45, f"End error too high for sentence {idx}: {end_err}"
                assert sent["confidence"] >= 0.65, sent
                if idx > 0:
                    assert spoken[idx - 1]["end"] <= sent["start"]

            # Verify HTTP Range request support on audio endpoint
            range_req = urllib.request.Request(
                base_url + f"/api/projects/{project_id}/audio",
                headers={"Range": "bytes=0-511"},
            )
            with urllib.request.urlopen(range_req, timeout=5) as resp:
                assert resp.status == 206, f"Expected 206 Partial Content, got {resp.status}"
                chunk = resp.read()
                assert len(chunk) == 512

            # Verify PATCH / split / merge / export (with Chinese title!)
            first_id = spoken[0]["id"]
            patch_req = urllib.request.Request(
                base_url + f"/api/projects/{project_id}/sentences/{first_id}",
                data=json.dumps({"start": spoken[0]["start"] + 0.05}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PATCH",
            )
            with urllib.request.urlopen(patch_req, timeout=5) as resp:
                assert resp.status == 200

            for fmt in ("lrc", "srt", "json"):
                with urllib.request.urlopen(base_url + f"/api/projects/{project_id}/export?format={fmt}", timeout=5) as resp:
                    assert resp.status == 200
                    txt = resp.read().decode("utf-8")
                    assert "Westgate Sports Centre" in txt

            split_req = urllib.request.Request(
                base_url + f"/api/projects/{project_id}/sentences/{first_id}/split",
                data=json.dumps({"split_index": 13}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(split_req, timeout=5) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                assert sum(1 for x in res["sentences"] if x["kind"] != "header") == 7

            merge_req = urllib.request.Request(
                base_url + f"/api/projects/{project_id}/sentences/{first_id}/merge-next",
                data=b"",
                method="POST",
            )
            with urllib.request.urlopen(merge_req, timeout=5) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                assert sum(1 for x in res["sentences"] if x["kind"] != "header") == 6

        finally:
            if project_id:
                try:
                    del_req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/projects/{project_id}",
                        method="DELETE",
                    )
                    urllib.request.urlopen(del_req, timeout=3).read()
                except Exception:
                    pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_direct_alignment_accuracy() -> None:
    intro = "Cambridge practice test one. Part one."
    spoken_lines = [
        "Good morning, Westgate Sports Centre.",
        "Hello, I would like to book a tennis court for Saturday.",
        "Certainly, what time would you prefer?",
        "Around ten thirty in the morning if possible.",
        "That costs twenty five dollars per hour.",
        "Great, please reserve it under the name David Brown.",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mp3_path = tmp_dir / "direct_demo.mp3"
        pdf_path = tmp_dir / "direct_demo.pdf"

        gt_spans = _synthesize_spoken_mp3([intro] + spoken_lines, mp3_path)
        sentence_gt = gt_spans[1:]

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 40), "Listening Transcript Demo", fontsize=10)
        page.insert_text((50, 95), "PART 1", fontsize=13)
        for idx, line in enumerate(spoken_lines):
            speaker = "RECEPTIONIST" if idx % 2 == 0 else "CALLER"
            page.insert_text((50, 130 + idx * 35), f"{speaker}: {line}", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        items = parse_pdf_transcript(str(pdf_path))
        res = align_audio_and_transcript(str(mp3_path), str(tmp_dir), items, model_size="base", language="en")
        spoken = [s for s in res["sentences"] if s["kind"] != "header"]
        assert len(spoken) == 6, spoken
        for idx, (sent, (gt_s, gt_e)) in enumerate(zip(spoken, sentence_gt)):
            start_err = abs(sent["start"] - gt_s)
            end_err = abs(sent["end"] - gt_e)
            print(
                f"  [Direct] Sentence {idx+1}: aligned=[{sent['start']:.2f}, {sent['end']:.2f}] "
                f"gt=[{gt_s:.2f}, {gt_e:.2f}] err=({start_err:.2f}s, {end_err:.2f}s) conf={sent['confidence']}"
            )
            assert start_err <= 0.35, f"Start error too high for sentence {idx+1}: {start_err}"
            assert end_err <= 0.35, f"End error too high for sentence {idx+1}: {end_err}"
            assert sent["confidence"] >= 0.70, sent
            if idx > 0:
                assert spoken[idx - 1]["end"] <= sent["start"]


if __name__ == "__main__":
    print("Running test_sentence_splitting_and_abbreviations...")
    test_sentence_splitting_and_abbreviations()
    print("Running test_pdf_parser_headers_columns_and_speakers...")
    test_pdf_parser_headers_columns_and_speakers()
    print("Running test_normalization_and_dp_alignment...")
    test_normalization_and_dp_alignment()
    print("Running test_direct_alignment_accuracy...")
    test_direct_alignment_accuracy()
    if "--no-server" not in sys.argv:
        print("Running test_end_to_end_server_and_alignment...")
        test_end_to_end_server_and_alignment()
    print("ALL TESTS PASSED!")
