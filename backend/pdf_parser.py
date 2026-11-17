"""PDF and plain-text transcript extraction with smart sentence segmentation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - fallback for older PyMuPDF releases
    import fitz


_PAGE_NUM_RE = re.compile(
    r"^\s*(?:[-—–•|]\s*)?(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?(?:\s*[-—–•|])?\s*$",
    re.IGNORECASE,
)

_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:part|section|test|passage|track|recording|unit|lesson|chapter|module|episode|dialogue|conversation)\s*(?:\d+|[ivx]+|[a-z])?(?:\s*[:\-–—.].*)?"
    r"|(?:.*?\b(?:audioscript|tapescript|listening\s+script|transcript)\b[^.!?]*)"
    r"|questions?\s+\d+\s*[-–—to]+\s*\d+"
    r"|第[一二三四五六七八九十\d]+[部分节章课篇题]"
    r")$",
    re.IGNORECASE,
)

# Inline speaker labels like "MAN:", "Receptionist:", "Speaker 1:", "Dr. Lee:", "男："
_SPEAKER_INLINE_RE = re.compile(
    r"^([A-Z][A-Z0-9 .'\-]{0,24}|[A-Z][a-z]+(?:\s+[A-Z0-9][a-zA-Z0-9.'\-]*){0,2}|[\u4e00-\u9fff]{1,6})\s*[:：]\s*(.+)$"
)

# Standalone speaker line in ALL CAPS or explicit role, e.g. "RECEPTIONIST" or "SPEAKER 1"
_SPEAKER_SOLO_WORDS = {
    "MAN", "WOMAN", "BOY", "GIRL", "HOST", "GUEST", "NARRATOR", "INTERVIEWER",
    "INTERVIEWEE", "RECEPTIONIST", "CUSTOMER", "CLIENT", "CALLER", "AGENT",
    "CLERK", "DOCTOR", "NURSE", "PATIENT", "TEACHER", "TUTOR", "PROFESSOR",
    "STUDENT", "EXAMINER", "CANDIDATE", "ANNOUNCER", "PRESENTER", "REPORTER",
    "SPEAKER", "MANAGER", "ASSISTANT", "GUIDE", "OFFICER", "LIBRARIAN",
}
_SPEAKER_SOLO_RE = re.compile(
    r"^(?:[A-Z]{2,18}(?:\s+(?:\d+|[A-Z]))?|(?:Speaker|Man|Woman|Host|Guest|Interviewer|Narrator|Student|Tutor|Doctor|Receptionist)\s*(?:\d+|[A-Z])?)\s*[:：]?$"
)

# Non-speaker words that commonly appear before a colon in prose
_NON_SPEAKER_PREFIXES = {
    "note", "notes", "tip", "tips", "example", "examples", "answer", "answers",
    "question", "questions", "warning", "hint", "summary", "reason", "reasons",
    "step", "steps", "time", "date", "place", "topic", "title", "address",
    "tel", "phone", "email", "website", "url", "http", "https", "price", "cost",
    "hours", "open", "closed", "nb", "ps", "attention", "important",
}

_ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "No.", "Fig.",
    "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "A.M.", "P.M.",
    "U.S.", "U.K.", "U.N.", "E.U.", "Jan.", "Feb.", "Mar.", "Apr.",
    "Aug.", "Sept.", "Sep.", "Oct.", "Nov.", "Dec.", "Inc.", "Ltd.", "Co.",
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _is_cjk(ch: str) -> bool:
    return bool(ch and _CJK_RE.match(ch))


def _join_text_fragments(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    # Rejoin hyphenated word broken across PDF lines: "conver-" + "sation"
    if left.endswith("-") and len(left) >= 2 and left[-2].isalpha() and right[0].islower():
        return left[:-1] + right
    # No space between adjacent CJK characters
    if _is_cjk(left[-1]) and _is_cjk(right[0]):
        return left + right
    return left + " " + right


def _is_valid_speaker_label(label: str) -> bool:
    clean = label.strip()
    if not clean or len(clean) > 28:
        return False
    low = clean.lower()
    if low in _NON_SPEAKER_PREFIXES:
        return False
    # Reject things like "10:30" or numbers
    if re.fullmatch(r"[\d\s.\-]+", clean):
        return False
    words = clean.split()
    if len(words) > 3:
        return False
    # If all uppercase or known role or CJK 1-6 chars or single capital letter A/B/M/W
    if clean.isupper() or len(clean) == 1 or _is_cjk(clean[0]):
        return True
    # Title Case (e.g., "Receptionist", "Dr. Smith", "Speaker 1")
    if all(w[0].isupper() or w.isdigit() for w in words if w):
        return True
    return False


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Replace abbreviations and decimals with placeholders before splitting."""
    placeholders: dict[str, str] = {}
    counter = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"__ABBR{counter}__"
        counter += 1
        placeholders[token] = match.group(0)
        return token

    # Ellipsis
    text = re.sub(r"\.{3,}|…+", _sub, text)
    # Decimal numbers or version numbers like 3.14, £12.50, 1.2.3
    text = re.sub(r"\d+(?:\.\d+)+", _sub, text)
    # Initialisms like U.S.A., Ph.D., a.m.
    text = re.sub(r"\b(?:[A-Za-z]\.){2,}", _sub, text)
    # Known abbreviations
    for abbr in _ABBREVIATIONS:
        escaped = re.escape(abbr)
        text = re.sub(rf"(?<!\w){escaped}", _sub, text)
    return text, placeholders


def _restore_abbreviations(text: str, placeholders: dict[str, str]) -> str:
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


def split_sentences(text: str) -> list[str]:
    """Split a paragraph/turn into clean sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected, placeholders = _protect_abbreviations(text)

    # Split after Latin terminal punctuation + closing quotes/brackets + whitespace
    # OR after CJK terminal punctuation (with optional closing quote/bracket)
    pattern = re.compile(
        r'(?<=[.!?])(?:["\'”’)\]]*)(?=\s+(?:[A-Z0-9"\'“‘(\[]|[\u4e00-\u9fff\u3040-\u30ff]))'
        r'|(?<=[。！？])(?:["\'”’）」』】\]]*)'
    )
    parts = pattern.split(protected)
    out: list[str] = []
    for part in parts:
        restored = _restore_abbreviations(part, placeholders).strip()
        if restored:
            # Merge stray punctuation-only fragments into previous sentence
            if out and re.fullmatch(r"[\"'”’)\].!?,;:。！？，；：]+", restored):
                out[-1] = out[-1] + restored
            else:
                out.append(restored)

    # If a sentence is extremely long (>320 chars) with semicolons, split gently on semicolons
    final: list[str] = []
    for s in out:
        if len(s) > 320 and ";" in s:
            subparts = [p.strip() for p in re.split(r"(?<=;)\s+", s) if p.strip()]
            final.extend(subparts)
        else:
            final.append(s)
    return final


def _sort_page_blocks(blocks: list[tuple[Any, ...]], page_width: float) -> list[tuple[Any, ...]]:
    """Sort blocks in reading order, handling single-column and two-column layouts."""
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if len(text_blocks) < 4 or page_width <= 0:
        return sorted(text_blocks, key=lambda b: (round(b[1] / 6.0), b[0]))

    mid = page_width * 0.5
    tol = page_width * 0.03
    left = [b for b in text_blocks if b[2] <= mid + tol]
    right = [b for b in text_blocks if b[0] >= mid - tol]
    spanning = [b for b in text_blocks if b[0] < mid - tol and b[2] > mid + tol]

    if len(left) >= 2 and len(right) >= 2 and len(spanning) <= max(2, len(text_blocks) // 4):
        col_top = min(b[1] for b in left + right)
        top_span = sorted([b for b in spanning if b[3] <= col_top + 10], key=lambda b: (b[1], b[0]))
        bot_span = sorted([b for b in spanning if b[3] > col_top + 10], key=lambda b: (b[1], b[0]))
        left_sorted = sorted(left, key=lambda b: (b[1], b[0]))
        right_sorted = sorted(right, key=lambda b: (b[1], b[0]))
        return top_span + left_sorted + right_sorted + bot_span

    return sorted(text_blocks, key=lambda b: (round(b[1] / 6.0), b[0]))


def extract_lines_from_pdf(pdf_path: str) -> list[str]:
    """Extract clean ordered text lines from a PDF, removing headers, footers, and page numbers."""
    doc = fitz.open(pdf_path)
    page_items: list[list[tuple[str, float, float, float]]] = []
    edge_counter: Counter[str] = Counter()

    try:
        num_pages = len(doc)
        for page in doc:
            h = float(page.rect.height) or 800.0
            w = float(page.rect.width) or 600.0
            blocks = _sort_page_blocks(page.get_text("blocks"), w)
            lines_on_page: list[tuple[str, float, float, float]] = []
            seen_edge_on_page: set[str] = set()

            for b in blocks:
                x0, y0, x1, y1, raw_text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), b[4]
                raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                if not raw_lines:
                    continue
                is_edge = y0 < h * 0.085 or y1 > h * 0.915
                for idx, ln in enumerate(raw_lines):
                    # Normalize digits inside repeated running headers like "Test 1  Page 3"
                    norm_edge = re.sub(r"\d+", "#", ln.lower().strip())
                    if is_edge and norm_edge not in seen_edge_on_page:
                        edge_counter[norm_edge] += 1
                        seen_edge_on_page.add(norm_edge)
                    lines_on_page.append((ln, y0 if idx == 0 else -1.0, y1, h))
            page_items.append(lines_on_page)
    finally:
        doc.close()

    repeated_edges = {k for k, cnt in edge_counter.items() if num_pages >= 2 and cnt >= 2}

    clean_lines: list[str] = []
    for lines_on_page in page_items:
        for ln, y0, y1, h in lines_on_page:
            if _PAGE_NUM_RE.match(ln):
                continue
            if ln.startswith("©") and len(ln) < 120:
                continue
            norm_edge = re.sub(r"\d+", "#", ln.lower().strip())
            if (y0 >= 0 and y0 < h * 0.085 or y1 > h * 0.915) and norm_edge in repeated_edges:
                continue
            clean_lines.append(ln)

        # Insert paragraph boundary marker between pages if previous line ended cleanly
        if clean_lines and clean_lines[-1] != "":
            if clean_lines[-1][-1] in ".!?。！？\"'”’":
                clean_lines.append("")

    return clean_lines


def parse_transcript_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Turn extracted transcript lines into structured header + sentence entries."""
    items: list[dict[str, Any]] = []
    current_speaker = ""
    current_section = ""
    buffer = ""
    pending_solo_speaker = ""

    def flush_buffer() -> None:
        nonlocal buffer
        text = buffer.strip()
        buffer = ""
        if not text:
            return
        for sent in split_sentences(text):
            items.append(
                {
                    "id": len(items),
                    "kind": "sentence",
                    "text": sent,
                    "speaker": current_speaker,
                    "section": current_section,
                    "start": 0.0,
                    "end": 0.0,
                    "confidence": 0.0,
                }
            )

    for raw_ln in lines:
        ln = re.sub(r"\s+", " ", raw_ln).strip()
        if not ln:
            flush_buffer()
            continue

        # Skip stage directions like [Music] or (Pause) alone on a line
        if re.fullmatch(r"[\[(][A-Za-z0-9\s,\-–—]{1,40}[\])]", ln):
            continue

        # Check section headers like "PART 1", "SECTION 2", "Audioscript"
        if len(ln) <= 70 and _SECTION_HEADER_RE.match(ln):
            flush_buffer()
            pending_solo_speaker = ""
            current_speaker = ""
            current_section = ln
            items.append(
                {
                    "id": len(items),
                    "kind": "header",
                    "text": ln,
                    "speaker": "",
                    "section": ln,
                    "start": 0.0,
                    "end": 0.0,
                    "confidence": 1.0,
                }
            )
            continue

        # Check standalone speaker label on its own line (e.g. "RECEPTIONIST" or "MAN:")
        if len(ln) <= 30 and _SPEAKER_SOLO_RE.match(ln):
            candidate = ln.rstrip(":： ").strip()
            if candidate.upper().split()[0] in _SPEAKER_SOLO_WORDS or ln.endswith((":", "：")):
                flush_buffer()
                current_speaker = candidate
                pending_solo_speaker = candidate
                continue

        # Check inline speaker label like "MAN: Hello there."
        m = _SPEAKER_INLINE_RE.match(ln)
        if m and _is_valid_speaker_label(m.group(1)):
            flush_buffer()
            pending_solo_speaker = ""
            current_speaker = m.group(1).strip()
            body = m.group(2).strip()
            buffer = body
            continue

        if pending_solo_speaker:
            pending_solo_speaker = ""
            buffer = ln
            continue

        # Join wrapped line into current paragraph buffer
        if not buffer:
            buffer = ln
        else:
            buffer = _join_text_fragments(buffer, ln)

    flush_buffer()
    return items


def parse_pdf_transcript(pdf_path: str) -> list[dict[str, Any]]:
    """Extract and segment a listening transcript PDF into sentence items."""
    lines = extract_lines_from_pdf(pdf_path)
    return parse_transcript_lines(lines)


def parse_plain_transcript(text: str) -> list[dict[str, Any]]:
    """Parse plain text / pasted transcript into sentence items."""
    lines = [ln.strip() for ln in text.splitlines()]
    return parse_transcript_lines(lines)
