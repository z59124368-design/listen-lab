"""Audio decoding, Whisper word-timestamp extraction, DP forced alignment, and RMS boundary snapping."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
_ORDINALS = {
    "1st": ["first"], "2nd": ["second"], "3rd": ["third"],
    "4th": ["fourth"], "5th": ["fifth"], "6th": ["sixth"],
    "7th": ["seventh"], "8th": ["eighth"], "9th": ["ninth"],
    "10th": ["tenth"], "11th": ["eleventh"], "12th": ["twelfth"],
    "13th": ["thirteenth"], "14th": ["fourteenth"], "15th": ["fifteenth"],
    "16th": ["sixteenth"], "17th": ["seventeenth"], "18th": ["eighteenth"],
    "19th": ["nineteenth"], "20th": ["twentieth"], "21st": ["twenty", "first"],
    "22nd": ["twenty", "second"], "23rd": ["twenty", "third"],
    "30th": ["thirtieth"], "31st": ["thirty", "first"],
}
_CONTRACTIONS = {
    "i'm": ["i", "am"],
    "i've": ["i", "have"],
    "i'll": ["i", "will"],
    "i'd": ["i", "would"],
    "you're": ["you", "are"],
    "you've": ["you", "have"],
    "you'll": ["you", "will"],
    "you'd": ["you", "would"],
    "he's": ["he", "is"],
    "she's": ["she", "is"],
    "it's": ["it", "is"],
    "we're": ["we", "are"],
    "we've": ["we", "have"],
    "we'll": ["we", "will"],
    "we'd": ["we", "would"],
    "they're": ["they", "are"],
    "they've": ["they", "have"],
    "they'll": ["they", "will"],
    "they'd": ["they", "would"],
    "that's": ["that", "is"],
    "there's": ["there", "is"],
    "here's": ["here", "is"],
    "what's": ["what", "is"],
    "who's": ["who", "is"],
    "where's": ["where", "is"],
    "let's": ["let", "us"],
    "can't": ["can", "not"],
    "cannot": ["can", "not"],
    "won't": ["will", "not"],
    "don't": ["do", "not"],
    "doesn't": ["does", "not"],
    "didn't": ["did", "not"],
    "isn't": ["is", "not"],
    "aren't": ["are", "not"],
    "wasn't": ["was", "not"],
    "weren't": ["were", "not"],
    "haven't": ["have", "not"],
    "hasn't": ["has", "not"],
    "hadn't": ["had", "not"],
    "couldn't": ["could", "not"],
    "wouldn't": ["would", "not"],
    "shouldn't": ["should", "not"],
    "mustn't": ["must", "not"],
    "needn't": ["need", "not"],
}
_COMMON_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "is", "are",
    "was", "were", "be", "it", "i", "you", "he", "she", "we", "they", "that",
    "this", "for", "with", "as", "by", "do", "not", "have", "has", "had",
}

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_TOKEN_RE = re.compile(
    r"[$£€¥]?\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th|%)?"
    r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
    r"|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)


@dataclass
class TimedToken:
    text: str
    start: float
    end: float


def _int_to_words(n: int) -> list[str]:
    if n < 0:
        return ["minus"] + _int_to_words(-n)
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        tens, rem = divmod(n, 10)
        return [_TENS[tens * 10]] + ([_ONES[rem]] if rem else [])
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        out = [_ONES[hundreds], "hundred"]
        if rem:
            out.extend(_int_to_words(rem))
        return out
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        out = _int_to_words(thousands) + ["thousand"]
        if rem:
            out.extend(_int_to_words(rem))
        return out
    return [str(n)]


def _expand_number_token(tok: str) -> list[str]:
    low = tok.lower().replace(",", "")
    if low in _ORDINALS:
        return list(_ORDINALS[low])

    currency_word = None
    if low.startswith("$"):
        currency_word = "dollars"
        low = low[1:]
    elif low.startswith("£"):
        currency_word = "pounds"
        low = low[1:]
    elif low.startswith("€"):
        currency_word = "euros"
        low = low[1:]

    pct = False
    if low.endswith("%"):
        pct = True
        low = low[:-1]

    if re.fullmatch(r"\d+\.\d+", low):
        whole, frac = low.split(".", 1)
        words = _int_to_words(int(whole))
        if currency_word:
            words.append(currency_word)
            if int(frac) > 0:
                words.extend(_int_to_words(int(frac)))
        else:
            words.append("point")
            words.extend(_ONES[int(d)] for d in frac)
        if pct:
            words.append("percent")
        return words

    if re.fullmatch(r"\d+", low):
        val = int(low)
        # Year pronunciation heuristic (e.g. 1998 -> nineteen ninety eight, 2024 -> twenty twenty four)
        if not currency_word and not pct and 1900 <= val <= 2099:
            hi, lo = divmod(val, 100)
            if lo == 0:
                words = _int_to_words(val)
            elif 2000 <= val <= 2009:
                words = ["two", "thousand"] + _int_to_words(lo)
            else:
                words = _int_to_words(hi) + ([_ONES[lo]] if lo < 10 else _int_to_words(lo))
        else:
            words = _int_to_words(val)
        if currency_word:
            words.append(currency_word)
        if pct:
            words.append("percent")
        return words

    cleaned = re.sub(r"[^a-z0-9]", "", low)
    return [cleaned] if cleaned else []


def normalize_text_to_tokens(text: str) -> list[str]:
    """Normalize arbitrary text into comparable word/CJK-char tokens."""
    text = text.replace("’", "'").replace("‘", "'")
    # Handle time like 9:30 -> 9 30
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)
    raw_matches = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for raw in raw_matches:
        low = raw.lower()
        if _CJK_CHAR_RE.match(low):
            tokens.append(low)
            continue
        if low in _CONTRACTIONS:
            tokens.extend(_CONTRACTIONS[low])
            continue
        if any(c.isdigit() for c in low):
            tokens.extend(_expand_number_token(low))
            continue
        if "'" in low:
            # e.g. "john's" -> "john", "s" (strip possessive/apostrophe cleanly)
            parts = [p for p in low.split("'") if p]
            if len(parts) == 2 and parts[1] == "s":
                tokens.append(parts[0] + "s")
            else:
                tokens.extend(parts)
            continue
        tokens.append(low)
    return tokens


def convert_to_wav_16k(audio_path: str, wav_path: str) -> None:
    """Convert any input audio file to 16kHz mono PCM WAV."""
    afconvert = "/usr/bin/afconvert"
    if os.path.exists(afconvert):
        proc = subprocess.run(
            [afconvert, audio_path, "-d", "LEI16@16000", "-c", "1", "-f", "WAVE", wav_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
            return

    # Fallback via PyAV (installed alongside faster-whisper)
    import av  # type: ignore

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(audio_path) as in_container:
        stream = next((s for s in in_container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("音频文件中未找到音频流")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for frame in in_container.decode(stream):
                for out_frame in resampler.resample(frame):
                    arr = out_frame.to_ndarray()
                    wf.writeframes(arr.astype("<i2", copy=False).tobytes())


def load_wav_and_features(
    wav_path: str, num_waveform_bars: int = 900
) -> tuple[float, list[float], np.ndarray, np.ndarray]:
    """Read 16kHz mono WAV and compute duration, UI waveform peaks, 10ms RMS envelope, and float32 samples."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    duration = float(len(samples)) / float(sr) if sr > 0 else 0.0
    if len(samples) == 0:
        return 0.0, [0.0] * num_waveform_bars, np.zeros(1, dtype=np.float32), samples

    # Compute waveform bars for UI display
    bars = min(num_waveform_bars, max(100, len(samples) // 160))
    chunk = max(1, len(samples) // bars)
    usable = (len(samples) // chunk) * chunk
    trimmed = np.abs(samples[:usable]).reshape(-1, chunk)
    # 92nd percentile per bucket looks cleaner and closer to perceived loudness than raw spikes
    peaks = np.percentile(trimmed, 92, axis=1)
    max_peak = float(np.max(peaks)) or 1.0
    norm_peaks = [round(float(v / max_peak), 3) for v in peaks]

    # Compute 10ms hop RMS envelope (hop=160 samples at 16kHz)
    hop = max(1, sr // 100)
    n_hops = len(samples) // hop
    if n_hops == 0:
        rms = np.array([float(np.sqrt(np.mean(samples ** 2)))], dtype=np.float32)
    else:
        framed = samples[: n_hops * hop].reshape(n_hops, hop)
        rms = np.sqrt(np.mean(framed * framed, axis=1)).astype(np.float32)

    return duration, norm_peaks, rms, samples


def _get_whisper_model(model_size: str) -> Any:
    allowed = {"tiny", "base", "small", "medium"}
    size = model_size if model_size in allowed else "base"
    with _MODEL_LOCK:
        if size in _MODEL_CACHE:
            return _MODEL_CACHE[size]

        from faster_whisper import WhisperModel

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        has_local = any(MODELS_DIR.glob(f"models--*faster-whisper-{size}*/snapshots/*/model.bin"))
        if has_local:
            try:
                model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODELS_DIR),
                    local_files_only=True,
                )
                _MODEL_CACHE[size] = model
                return model
            except Exception:
                pass

        try:
            model = WhisperModel(
                size,
                device="cpu",
                compute_type="int8",
                download_root=str(MODELS_DIR),
            )
        except Exception as first_err:
            # Retry using HuggingFace mirror (must patch huggingface_hub.constants.ENDPOINT too)
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                try:
                    import huggingface_hub.constants as hf_const
                    hf_const.ENDPOINT = "https://hf-mirror.com"
                except Exception:
                    pass
                model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODELS_DIR),
                )
            else:
                raise first_err

        _MODEL_CACHE[size] = model
        return model


def transcribe_word_tokens(
    audio_samples: np.ndarray,
    duration: float,
    model_size: str = "base",
    language: str = "auto",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> tuple[list[TimedToken], str]:
    """Run Whisper with word timestamps and return normalized TimedTokens + detected language."""
    if on_progress:
        on_progress(18, f"正在加载语音模型 ({model_size})...")
    model = _get_whisper_model(model_size)

    lang_arg = None if not language or language == "auto" else language
    if on_progress:
        on_progress(25, "正在识别音频语音与词级时间戳...")

    segments, info = model.transcribe(
        audio_samples,
        language=lang_arg,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 150},
    )

    raw_words: list[TimedToken] = []
    last_pct = 25
    for seg in segments:
        if seg.words:
            for w in seg.words:
                txt = (w.word or "").strip()
                if not txt:
                    continue
                ws = float(w.start)
                we = max(ws + 0.02, float(w.end))
                # Merge BPE apostrophe suffixes like "'t", "'m", "'s", "'re", "'ve", "'ll", "'d"
                if raw_words and (txt.startswith("'") or txt.startswith("’")) and len(txt) <= 4:
                    prev = raw_words[-1]
                    prev.text = prev.text + txt
                    prev.end = max(prev.end, we)
                else:
                    raw_words.append(TimedToken(text=txt, start=ws, end=we))
        else:
            txt = (seg.text or "").strip()
            if txt:
                raw_words.append(TimedToken(text=txt, start=float(seg.start), end=float(seg.end)))

        if on_progress and duration > 0:
            pct = 25 + int(55 * min(1.0, float(seg.end) / duration))
            if pct > last_pct + 2:
                last_pct = pct
                on_progress(pct, f"语音识别与时间轴对齐中 ({int(100 * min(1.0, seg.end / duration))}%)...")

    # Expand each raw ASR word into normalized tokens with proportional sub-timestamps
    asr_tokens: list[TimedToken] = []
    for rw in raw_words:
        sub = normalize_text_to_tokens(rw.text)
        if not sub:
            continue
        if len(sub) == 1:
            asr_tokens.append(TimedToken(text=sub[0], start=rw.start, end=rw.end))
        else:
            span = max(0.04, rw.end - rw.start)
            step = span / len(sub)
            for idx, tok in enumerate(sub):
                s_t = rw.start + idx * step
                e_t = rw.start + (idx + 1) * step
                asr_tokens.append(TimedToken(text=tok, start=s_t, end=e_t))

    detected_lang = getattr(info, "language", None) or (lang_arg or "en")
    return asr_tokens, detected_lang


def _word_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.6:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _match_score(p: str, a: str) -> float:
    if p == a:
        bonus = 0.6 if (len(p) >= 5 and p not in _COMMON_WORDS) else 0.0
        return 4.0 + bonus
    if len(p) == 1 and len(a) == 1:
        return -2.2
    sim = _word_similarity(p, a)
    if sim >= 0.72:
        return 2.6 * sim
    return -2.0


def _find_monotonic_anchors(p_tokens: list[str], a_tokens: list[str]) -> list[tuple[int, int]]:
    """Find monotonic 2-gram anchors between PDF tokens and ASR tokens via LIS."""
    n = len(p_tokens)
    k = len(a_tokens)
    if n < 20 or k < 20:
        return []

    a_bigrams: dict[tuple[str, str], list[int]] = {}
    for j in range(k - 1):
        bg = (a_tokens[j], a_tokens[j + 1])
        if bg[0] in _COMMON_WORDS and bg[1] in _COMMON_WORDS:
            continue
        lst = a_bigrams.setdefault(bg, [])
        if len(lst) <= 2:
            lst.append(j)

    p_counts: dict[tuple[str, str], int] = {}
    for i in range(n - 1):
        bg = (p_tokens[i], p_tokens[i + 1])
        p_counts[bg] = p_counts.get(bg, 0) + 1

    candidates: list[tuple[int, int]] = []
    for i in range(n - 1):
        bg = (p_tokens[i], p_tokens[i + 1])
        if p_counts.get(bg, 0) != 1:
            continue
        locs = a_bigrams.get(bg)
        if locs and len(locs) == 1:
            candidates.append((i, locs[0]))

    if len(candidates) < 2:
        return []

    # Longest increasing subsequence on j (since i is strictly increasing)
    tails: list[int] = []
    tail_idx: list[int] = []
    prev_ptr = [-1] * len(candidates)

    import bisect

    for idx, (_, j_val) in enumerate(candidates):
        pos = bisect.bisect_left(tails, j_val)
        if pos == len(tails):
            tails.append(j_val)
            tail_idx.append(idx)
        else:
            tails[pos] = j_val
            tail_idx[pos] = idx
        if pos > 0:
            prev_ptr[idx] = tail_idx[pos - 1]

    lis: list[tuple[int, int]] = []
    cur = tail_idx[-1]
    while cur != -1:
        lis.append(candidates[cur])
        cur = prev_ptr[cur]
    lis.reverse()

    # Keep anchors separated by >= 8 tokens so sub-intervals stay crisp
    filtered: list[tuple[int, int]] = []
    for pi, aj in lis:
        if not filtered or (pi - filtered[-1][0] >= 8 and aj - filtered[-1][1] >= 8):
            filtered.append((pi, aj))
    return filtered


def _dp_align_segment(
    p_slice: list[str],
    a_slice: list[str],
    free_leading_a: bool = False,
    free_trailing_a: bool = False,
) -> list[Optional[int]]:
    """Semi-global Needleman-Wunsch alignment for a slice; returns indices into a_slice."""
    n = len(p_slice)
    m = len(a_slice)
    if n == 0:
        return []
    if m == 0:
        return [None] * n

    skip_p_pen = -1.2
    skip_a_pen = -0.25

    # Banded if huge subproblem to guarantee fast execution
    use_band = (n * m) > 90_000
    band = max(120, abs(m - n) + 80) if use_band else m + 5
    neg_inf = -1e9

    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1: diag, 2: up (skip p), 3: left (skip a)

    dp[0][0] = 0.0
    for j in range(1, m + 1):
        if free_leading_a:
            dp[0][j] = 0.0
        else:
            dp[0][j] = dp[0][j - 1] + skip_a_pen
        bt[0][j] = 3

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + skip_p_pen
        bt[i][0] = 2

        center = int(round(i * (m / max(1, n))))
        j_lo = max(1, center - band) if use_band else 1
        j_hi = min(m, center + band) if use_band else m

        p_tok = p_slice[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        bt_i = bt[i]

        for j in range(j_lo, j_hi + 1):
            diag = dp_prev[j - 1] + _match_score(p_tok, a_slice[j - 1])
            up = dp_prev[j] + skip_p_pen
            left = dp_i[j - 1] + skip_a_pen

            if diag >= up and diag >= left:
                dp_i[j] = diag
                bt_i[j] = 1
            elif up >= left:
                dp_i[j] = up
                bt_i[j] = 2
            else:
                dp_i[j] = left
                bt_i[j] = 3

    best_j = m
    if free_trailing_a:
        best_val = neg_inf
        for j in range(m + 1):
            if dp[n][j] >= best_val:
                best_val = dp[n][j]
                best_j = j

    mapping: list[Optional[int]] = [None] * n
    i, j = n, best_j
    while i > 0:
        move = bt[i][j]
        if move == 1 and j > 0:
            # Only record match if words actually match or are similar enough
            if _match_score(p_slice[i - 1], a_slice[j - 1]) > 0:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == 2 or j == 0:
            i -= 1
        else:
            j -= 1
    return mapping


def align_tokens(p_tokens: list[str], a_tokens: list[TimedToken]) -> list[Optional[int]]:
    """Align PDF tokens to ASR TimedTokens globally using anchor-guided semi-global DP."""
    a_strs = [t.text for t in a_tokens]
    n = len(p_tokens)
    k = len(a_strs)
    if n == 0 or k == 0:
        return [None] * n

    anchors = _find_monotonic_anchors(p_tokens, a_strs)
    if not anchors:
        return _dp_align_segment(p_tokens, a_strs, free_leading_a=True, free_trailing_a=True)

    result: list[Optional[int]] = [None] * n
    prev_p = 0
    prev_a = 0

    for idx, (ap, aa) in enumerate(anchors):
        sub_map = _dp_align_segment(
            p_tokens[prev_p:ap],
            a_strs[prev_a:aa],
            free_leading_a=(idx == 0),
            free_trailing_a=False,
        )
        for off, rel_a in enumerate(sub_map):
            if rel_a is not None:
                result[prev_p + off] = prev_a + rel_a

        # Anchor bigram matches directly
        result[ap] = aa
        if ap + 1 < n and aa + 1 < k:
            result[ap + 1] = aa + 1
            prev_p = ap + 2
            prev_a = aa + 2
        else:
            prev_p = ap + 1
            prev_a = aa + 1

    # Final tail segment
    tail_map = _dp_align_segment(
        p_tokens[prev_p:n],
        a_strs[prev_a:k],
        free_leading_a=False,
        free_trailing_a=True,
    )
    for off, rel_a in enumerate(tail_map):
        if rel_a is not None:
            result[prev_p + off] = prev_a + rel_a

    return result


def _snap_boundaries_with_rms(
    sentences: list[dict[str, Any]],
    rms: np.ndarray,
    duration: float,
) -> None:
    """Refine sentence start/end timestamps using 10ms RMS energy silence detection."""
    if len(rms) < 10 or duration <= 0:
        return

    spoken = [s for s in sentences if s.get("kind") != "header"]
    if not spoken:
        return

    noise_floor = float(np.percentile(rms, 15))
    peak = float(np.max(rms))
    thr = max(noise_floor * 2.6, peak * 0.025, 1e-4)
    hop_sec = 0.01

    def time_to_frame(t: float) -> int:
        return max(0, min(len(rms) - 1, int(round(t / hop_sec))))

    raw_starts = [s["start"] for s in spoken]
    raw_ends = [s["end"] for s in spoken]

    for idx, s in enumerate(spoken):
        rs = raw_starts[idx]
        re_t = raw_ends[idx]

        left_limit = 0.0 if idx == 0 else max(0.0, 0.5 * (raw_ends[idx - 1] + rs))
        right_limit = duration if idx == len(spoken) - 1 else min(duration, 0.5 * (re_t + raw_starts[idx + 1]))

        # 1. Refine start boundary
        sf = time_to_frame(rs)
        min_sf = time_to_frame(max(left_limit, rs - 0.25))
        if rms[sf] > thr:
            # Walk backward while still in active speech (tolerance of 1 dip frame)
            cur = sf
            while cur > min_sf and rms[cur - 1] > thr:
                cur -= 1
            new_start = max(left_limit, cur * hop_sec - 0.05)
        else:
            # Whisper often includes leading silence inside the first word; trim up to 1.5s forward
            max_sf = time_to_frame(min(re_t - 0.15, rs + 1.50))
            cur = sf
            while cur + 1 < max_sf and not (rms[cur] > thr and rms[cur + 1] > thr):
                cur += 1
            if cur + 1 < max_sf:
                new_start = max(left_limit, cur * hop_sec - 0.06)
            else:
                new_start = rs

        # 2. Refine end boundary
        ef = time_to_frame(re_t)
        if rms[ef] > thr:
            # Walk forward up to 0.30s if speech continues past Whisper's word end
            max_ef = time_to_frame(min(right_limit, re_t + 0.30))
            cur = ef
            silent_run = 0
            last_voice = ef
            while cur < max_ef:
                if rms[cur] > thr:
                    last_voice = cur
                    silent_run = 0
                else:
                    silent_run += 1
                    if silent_run >= 4:
                        break
                cur += 1
            new_end = min(right_limit, max(re_t, (last_voice + 1) * hop_sec + 0.08))
        else:
            # Check if Whisper padded trailing silence onto the last word (search back up to 1.2s)
            min_ef = time_to_frame(max(new_start + 0.20, re_t - 1.20))
            cur = ef
            while cur > min_ef and not (rms[cur] > thr and rms[cur - 1] > thr):
                cur -= 1
            if cur > min_ef:
                new_end = min(right_limit, (cur + 1) * hop_sec + 0.08)
            else:
                new_end = min(right_limit, re_t)

        s["start"] = round(max(0.0, new_start), 3)
        s["end"] = round(min(duration, max(s["start"] + 0.22, new_end)), 3)

    # Final strict monotonicity pass
    for idx in range(len(spoken) - 1):
        if spoken[idx]["end"] > spoken[idx + 1]["start"]:
            mid = round(0.5 * (spoken[idx]["end"] + spoken[idx + 1]["start"]), 3)
            spoken[idx]["end"] = max(spoken[idx]["start"] + 0.15, mid)
            spoken[idx + 1]["start"] = spoken[idx]["end"]
            spoken[idx + 1]["end"] = max(spoken[idx + 1]["start"] + 0.20, spoken[idx + 1]["end"])


def align_audio_and_transcript(
    audio_path: str,
    work_dir: str,
    items: list[dict[str, Any]],
    model_size: str = "base",
    language: str = "auto",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> dict[str, Any]:
    """Run full audio conversion, ASR, DP alignment, interpolation, and RMS boundary refinement."""
    wav_path = str(Path(work_dir) / "audio_16k.wav")
    if on_progress:
        on_progress(8, "正在解码音频并生成波形...")
    convert_to_wav_16k(audio_path, wav_path)

    duration, waveform_peaks, rms, samples = load_wav_and_features(wav_path)
    if duration <= 0.0:
        raise ValueError("音频时长无效或文件损坏")

    asr_tokens, detected_lang = transcribe_word_tokens(
        audio_samples=samples,
        duration=duration,
        model_size=model_size,
        language=language,
        on_progress=on_progress,
    )

    if on_progress:
        on_progress(85, "正在将 PDF 原文逐句对齐到音频时间轴...")

    spoken_indices = [idx for idx, it in enumerate(items) if it.get("kind") != "header"]
    p_tokens: list[str] = []
    sent_token_ranges: dict[int, tuple[int, int]] = {}

    for idx in spoken_indices:
        toks = normalize_text_to_tokens(items[idx]["text"])
        start_tok = len(p_tokens)
        p_tokens.extend(toks)
        sent_token_ranges[idx] = (start_tok, len(p_tokens))

    mapping = align_tokens(p_tokens, asr_tokens)

    # First pass: compute direct timestamps for anchored sentences
    anchored_idx: list[int] = []
    for idx in spoken_indices:
        t0, t1 = sent_token_ranges[idx]
        total_toks = max(1, t1 - t0)
        matched_asr = [mapping[p] for p in range(t0, t1) if mapping[p] is not None]
        if matched_asr:
            first_a = matched_asr[0]
            last_a = matched_asr[-1]
            # Look at unmatched head/tail words inside sentence
            first_matched_p = next(p for p in range(t0, t1) if mapping[p] is not None)
            last_matched_p = next(p for p in range(t1 - 1, t0 - 1, -1) if mapping[p] is not None)
            missing_head = first_matched_p - t0
            missing_tail = (t1 - 1) - last_matched_p

            s_time = asr_tokens[first_a].start
            if missing_head > 0 and first_a > 0:
                prev_tok_end = asr_tokens[first_a - 1].end
                if s_time - prev_tok_end < 0.8:
                    s_time = max(prev_tok_end, s_time - 0.18 * min(2, missing_head))

            e_time = asr_tokens[last_a].end
            if missing_tail > 0 and last_a + 1 < len(asr_tokens):
                next_tok_start = asr_tokens[last_a + 1].start
                if next_tok_start - e_time < 0.8:
                    e_time = min(next_tok_start, e_time + 0.18 * min(2, missing_tail))

            items[idx]["start"] = round(max(0.0, s_time), 3)
            items[idx]["end"] = round(min(duration, max(s_time + 0.25, e_time)), 3)
            items[idx]["confidence"] = round(min(1.0, len(matched_asr) / total_toks), 2)
            anchored_idx.append(idx)
        else:
            items[idx]["confidence"] = 0.0

    # Second pass: interpolate any unanchored spoken sentences cleanly between anchors
    if not anchored_idx and spoken_indices:
        # Fallback proportional split over full duration if ASR returned nothing
        weights = [max(4, len(items[i]["text"])) for i in spoken_indices]
        total_w = sum(weights)
        cur_t = 0.0
        for idx, w in zip(spoken_indices, weights):
            span = duration * (w / total_w)
            items[idx]["start"] = round(cur_t, 3)
            cur_t += span
            items[idx]["end"] = round(min(duration, cur_t), 3)
            items[idx]["confidence"] = 0.1
    else:
        # Walk spoken_indices to fill contiguous unanchored runs
        pos = 0
        while pos < len(spoken_indices):
            idx = spoken_indices[pos]
            if items[idx]["confidence"] > 0:
                pos += 1
                continue
            run_start = pos
            while pos < len(spoken_indices) and items[spoken_indices[pos]]["confidence"] == 0:
                pos += 1
            run_end = pos  # exclusive

            t_left = 0.0 if run_start == 0 else items[spoken_indices[run_start - 1]]["end"]
            t_right = duration if run_end >= len(spoken_indices) else items[spoken_indices[run_end]]["start"]
            if t_right <= t_left:
                t_right = min(duration, t_left + 0.5 * (run_end - run_start))

            weights = [max(4, len(items[spoken_indices[k]]["text"])) for k in range(run_start, run_end)]
            total_w = sum(weights)
            cur_t = t_left
            for k, w in zip(range(run_start, run_end), weights):
                s_idx = spoken_indices[k]
                span = (t_right - t_left) * (w / total_w)
                items[s_idx]["start"] = round(cur_t, 3)
                cur_t += span
                items[s_idx]["end"] = round(min(duration, max(items[s_idx]["start"] + 0.2, cur_t)), 3)
                items[s_idx]["confidence"] = 0.15

    # Ensure headers point to the start of the first sentence following them
    next_start = duration
    for idx in range(len(items) - 1, -1, -1):
        if items[idx].get("kind") == "header":
            items[idx]["start"] = round(next_start, 3)
            items[idx]["end"] = round(next_start, 3)
        else:
            next_start = items[idx]["start"]

    if on_progress:
        on_progress(94, "正在根据音频静音能量微调句首句尾边界...")

    _snap_boundaries_with_rms(items, rms, duration)

    # Remove temporary wav to keep project disk usage lean
    try:
        os.remove(wav_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, "对齐完成！")

    return {
        "duration": round(duration, 3),
        "detected_language": detected_lang,
        "waveform": waveform_peaks,
        "sentences": items,
    }
