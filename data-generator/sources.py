from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


PAGE_HEADING = re.compile(r"^## 인쇄 페이지 (\d+) \(PDF (\d+)\)$", re.MULTILINE)
PAGE_LINE = re.compile(r"^## 인쇄 페이지 (\d+) \(PDF \d+\)$")
TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def parse_pages(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    if not pages:
        raise ValueError("At least one printed page is required")
    return pages


def load_uvabot_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    if not rows or "Pregunta" not in rows[0]:
        raise ValueError("UVaBot CSV must contain the Pregunta column")
    return [
        {
            "source_id": f"uvabot-{index:04d}",
            "question": row["Pregunta"].strip(),
        }
        for index, row in enumerate(rows, start=1)
        if row.get("Pregunta", "").strip()
    ]


def _style_tags(question: str) -> set[str]:
    lowered = question.casefold()
    words = lowered.split()
    tags: set[str] = set()
    if len(words) <= 4:
        tags.add("fragment")
    if len(words) >= 25:
        tags.add("long_scenario")
    if "?" in question or "¿" in question:
        tags.add("explicit_question")
    if re.search(r"\b(hola|buenos días|buenas)\b", lowered):
        tags.add("greeting")
    if re.search(r"\b(yo|me|mi|estoy|tengo|quiero|he)\b", lowered):
        tags.add("first_person")
    if re.search(r"\b(cuando|cuándo|fecha|plazo)\b", lowered):
        tags.add("time")
    if re.search(r"\b(como|cómo|donde|dónde|que pasos)\b", lowered):
        tags.add("procedure")
    if re.search(r"\b(puedo|podría|se puede|hay que|tengo que)\b", lowered):
        tags.add("yes_no")
    if re.search(r"\bsi\b", lowered):
        tags.add("conditional")
    if not tags:
        tags.add("other")
    return tags


def select_diverse_questions(
    questions: list[dict[str, str]], limit: int = 48
) -> list[dict[str, str]]:
    if len(questions) <= limit:
        return questions
    buckets: dict[str, list[dict[str, str]]] = {}
    for question in questions:
        for tag in _style_tags(question["question"]):
            buckets.setdefault(tag, []).append(question)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    ordered_tags = [
        "fragment",
        "long_scenario",
        "greeting",
        "first_person",
        "time",
        "procedure",
        "yes_no",
        "conditional",
        "explicit_question",
        "other",
    ]
    cursor = 0
    while len(selected) < limit:
        added = False
        for tag in ordered_tags:
            bucket = buckets.get(tag, [])
            if cursor >= len(bucket):
                continue
            candidate = bucket[cursor]
            if candidate["source_id"] not in seen:
                selected.append(candidate)
                seen.add(candidate["source_id"])
                added = True
                if len(selected) == limit:
                    break
        if not added and all(cursor >= len(items) for items in buckets.values()):
            break
        cursor += 1

    if len(selected) < limit:
        for candidate in sorted(questions, key=lambda item: len(item["question"])):
            if candidate["source_id"] in seen:
                continue
            selected.append(candidate)
            seen.add(candidate["source_id"])
            if len(selected) == limit:
                break
    return selected


def load_hyu_context(
    preview_path: Path,
    notice_path: Path | None,
    pages: set[int] | None,
) -> str:
    markdown = preview_path.read_text(encoding="utf-8")
    matches = list(PAGE_HEADING.finditer(markdown))
    sections: list[str] = []
    for index, match in enumerate(matches):
        printed_page = int(match.group(1))
        if pages is not None and printed_page not in pages:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        section = markdown[match.start() : end].strip()
        if section:
            sections.append(section)

    if not sections:
        raise ValueError("No requested pages were found in the HYU preview")

    context_parts = [
        "# 문서 1: 2026학년도 2학기 한양대학교 서울캠퍼스 학사안내",
        *sections,
    ]
    if notice_path is not None and notice_path.exists():
        notice = json.loads(notice_path.read_text(encoding="utf-8"))
        context_parts.extend(
            [
                "# 문서 2: 정책학과 2026-2학기 수강신청 공지",
                f"- 문서 ID: {notice.get('document_id', 'policy-notice-2341')}",
                f"- 제목: {notice.get('title', '')}",
                f"- 게시 시각: {notice.get('published_at', '')}",
                notice.get("body", ""),
            ]
        )
    return "\n\n".join(context_parts).strip()


def extract_evidence_units(context: str) -> list[dict[str, str]]:
    """Split the rendered source into immutable, uniquely addressable evidence units."""
    lines = context.splitlines()
    units: list[dict[str, str]] = []
    seen: set[str] = set()
    locator = ""
    unit_index_by_locator: dict[str, int] = {}

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        page_match = PAGE_LINE.match(line)
        if page_match:
            locator = f"인쇄 페이지 {page_match.group(1)}"
            continue
        if line == "# 문서 2: 정책학과 2026-2학기 수강신청 공지":
            locator = "policy-notice-2341"
            continue
        if not locator or not line or line.startswith("#"):
            continue
        if line.startswith("- 문서 ID:") or line.startswith("- 게시 시각:"):
            continue
        if TABLE_SEPARATOR.match(line):
            continue
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if line.startswith("|") and TABLE_SEPARATOR.match(next_line):
            continue

        source_quote = re.sub(
            r"\|\s*(?:⚠️\s*검수 필요|✅\s*자동 승인)\s*\|$",
            "|",
            line,
        )
        prompt_content = re.sub(r"[⚠️✅]", "", source_quote)
        prompt_content = re.sub(r"^>\s*\*\*주석:\*\*\s*", "", prompt_content)

        visible = re.sub(r"[|>*_`⚠️✅—\-]", "", prompt_content)
        visible = re.sub(r"\s+", "", visible)
        if len(visible) < 20:
            continue

        normalized = re.sub(r"\s+", "", source_quote).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unit_index_by_locator[locator] = unit_index_by_locator.get(locator, 0) + 1
        ordinal = unit_index_by_locator[locator]
        digest = hashlib.sha256(
            f"{locator}\n{source_quote}".encode("utf-8")
        ).hexdigest()[:10]
        units.append(
            {
                "source_unit_id": f"evidence-{locator.replace(' ', '-')}-{ordinal:03d}-{digest}",
                "locator": locator,
                "quote": source_quote,
                "prompt_content": prompt_content,
            }
        )
    return units


def select_evidence_units(
    units: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Round-robin source locations so one dense page cannot dominate a run."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    buckets: dict[str, list[dict[str, str]]] = {}
    for unit in units:
        buckets.setdefault(unit["locator"], []).append(unit)

    selected: list[dict[str, str]] = []
    depth = 0
    locators = sorted(
        buckets,
        key=lambda item: (
            item == "policy-notice-2341",
            int(item.rsplit(" ", 1)[-1]) if item.startswith("인쇄 페이지 ") else 999,
        ),
    )
    while len(selected) < limit:
        added = False
        for item in locators:
            if depth >= len(buckets[item]):
                continue
            selected.append(buckets[item][depth])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        depth += 1
    if len(selected) < limit:
        raise ValueError(
            f"Only {len(selected)} unique evidence units are available; requested {limit}"
        )
    return selected
