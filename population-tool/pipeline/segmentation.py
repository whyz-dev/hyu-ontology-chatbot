"""검수된 Markdown 문서를 페이지 단위 ontology evidence로 변환한다."""

from __future__ import annotations

import re
from pathlib import Path

from adapters.storage import read_json, sha256_file, sha256_text, write_jsonl
from config import UNIT_SCHEMA_VERSION
from domain.models import EvidenceUnit

PAGE_SEPARATOR = "\n\n---\n\n"
PAGE_MARKER = re.compile(
    r"^<!-- pdf_page=(\d+) printed_page=([^ ]+) transcription=manual -->$"
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_DIVIDER_CELL = re.compile(r"^:?-{3,}:?$")
ADAPTER_VERSION = "manual-markdown-adapter-v2"


def _markdown_cells(line: str) -> list[str]:
    """이스케이프된 pipe를 보존하면서 Markdown 표 셀을 나눈다."""
    value = line.strip()
    if not value.startswith("|") or not value.endswith("|"):
        raise ValueError(f"Invalid Markdown table row: {line!r}")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            current.append(character)
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    return cells


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or not lines[index].lstrip().startswith("|"):
        return False
    try:
        divider = _markdown_cells(lines[index + 1])
        header = _markdown_cells(lines[index])
    except ValueError:
        return False
    return len(header) == len(divider) and all(
        TABLE_DIVIDER_CELL.fullmatch(cell.replace(" ", "")) for cell in divider
    )


def _update_section_path(stack: list[str], level: int, title: str) -> None:
    del stack[level - 1 :]
    while len(stack) < level - 1:
        stack.append("")
    stack.append(title)


def _section_text(path: list[str]) -> str:
    return " > ".join(value for value in path if value)


def _split_pages(document: str) -> list[str]:
    parts = document.rstrip("\n").split(PAGE_SEPARATOR)
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("Manual Markdown contains an empty page segment")
    return [part.rstrip() + "\n" for part in parts]


def _manual_manifest(input_path: Path) -> dict[str, object]:
    manifest_path = input_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manual manifest does not exist: {manifest_path}")
    value = read_json(manifest_path)
    if not isinstance(value, dict):
        raise TypeError(f"Manual manifest is not an object: {manifest_path}")
    outputs = value.get("outputs")
    aggregate = outputs.get("aggregate") if isinstance(outputs, dict) else None
    if not isinstance(aggregate, dict) or aggregate.get("sha256") != sha256_file(
        input_path
    ):
        raise ValueError("document.md does not match the manual manifest hash")
    return value


def _page_unit(
    page_text: str,
    document_id: str,
    document_scope: str,
    section_stack: list[str],
    source_limited_pages: set[int],
    previous_table: dict[str, object] | None,
) -> tuple[EvidenceUnit, dict[str, object] | None]:
    lines = page_text.rstrip("\n").splitlines()
    marker = PAGE_MARKER.fullmatch(lines[0]) if lines else None
    if marker is None:
        raise ValueError("Each manual page must start with a pdf_page marker")
    pdf_page = int(marker.group(1))
    printed_value = marker.group(2)
    printed_page = None if printed_value == "frontmatter" else int(printed_value)
    page_prefix = f"p{pdf_page:03d}"
    page_locator = f"page-{pdf_page:03d}"

    entry_section_path = [value for value in section_stack if value]
    rendered = [
        f"[PAGE {page_locator} | adapter={ADAPTER_VERSION}]",
        f"document={document_id} | printed_page={printed_page}",
        f"[DOCUMENT SCOPE] {document_scope}",
    ]
    if entry_section_path:
        rendered.append(f"[INHERITED SECTION] {_section_text(entry_section_path)}")
    locators: list[str] = []
    evidence: dict[str, dict[str, object]] = {}
    block_number = 0
    table_number = 0
    first_semantic_node = True
    last_node_kind: str | None = None
    last_table: dict[str, object] | None = None
    index = 1
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = HEADING.match(line)
        if heading:
            _update_section_path(
                section_stack,
                len(heading.group(1)),
                heading.group(2).strip(),
            )
            rendered.append(f"[SECTION] {_section_text(section_stack)}")
            first_semantic_node = False
            last_node_kind = "heading"
            last_table = None
            index += 1
            continue
        if _is_table_start(lines, index):
            table_number += 1
            table_id = f"{page_prefix}-table-{table_number:02d}"
            header_line = lines[index]
            physical_headers = _markdown_cells(header_line)
            section_path = [value for value in section_stack if value]
            continuation = bool(
                first_semantic_node
                and previous_table
                and all(not value for value in physical_headers)
                and len(physical_headers) == len(previous_table.get("headers", []))
                and section_path == previous_table.get("section_path")
            )
            headers = (
                list(previous_table["headers"])
                if continuation and previous_table is not None
                else physical_headers
            )
            rendered.append(
                f"[TABLE {table_id} | section={_section_text(section_path)}"
                + (
                    f" | continuation_of={previous_table['table_id']}]"
                    if continuation and previous_table is not None
                    else "]"
                )
            )
            header_locator = f"{table_id}-row-0000"
            locators.append(header_locator)
            evidence[header_locator] = {
                "kind": "table_header",
                "role": "header",
                "quote": header_line,
                "grounding_text": "\n".join(
                    [document_scope, *section_path, header_line]
                ),
                "section_path": section_path,
                "table_id": table_id,
                "header": header_line,
                "effective_headers": headers,
                "line_start": index + 1,
                "line_end": index + 1,
            }
            rendered.append(f"[ROW {header_locator} | role=header] {header_line}")
            index += 2
            row_number = 0
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                raw_row = lines[index]
                values = _markdown_cells(raw_row)
                if len(values) != len(headers):
                    raise ValueError(
                        f"Page {pdf_page} table {table_number} has inconsistent width"
                    )
                row_number += 1
                locator = f"{table_id}-row-{row_number:04d}"
                locators.append(locator)
                pairs = [
                    f"{header or f'column-{position + 1}'}={value or '—'}"
                    for position, (header, value) in enumerate(zip(headers, values))
                ]
                grounding = "\n".join(
                    [document_scope, *section_path, header_line, raw_row]
                )
                evidence[locator] = {
                    "kind": "table_row",
                    "quote": raw_row,
                    "grounding_text": grounding,
                    "section_path": section_path,
                    "table_id": table_id,
                    "header": header_line,
                    "effective_headers": headers,
                    "continuation_of": (
                        previous_table.get("table_id")
                        if continuation and previous_table is not None
                        else None
                    ),
                    "line_start": index + 1,
                    "line_end": index + 1,
                }
                rendered.append(f"[ROW {locator}] " + " | ".join(pairs))
                index += 1
            last_node_kind = "table"
            last_table = {
                "table_id": table_id,
                "headers": headers,
                "section_path": section_path,
            }
            first_semantic_node = False
            continue

        start = index
        block_lines = [line]
        index += 1
        while index < len(lines):
            if not lines[index].strip() or HEADING.match(lines[index]):
                break
            if _is_table_start(lines, index):
                break
            block_lines.append(lines[index])
            index += 1
        quote = "\n".join(block_lines).strip()
        if quote:
            block_number += 1
            locator = f"{page_prefix}-block-{block_number:04d}"
            section_path = [value for value in section_stack if value]
            locators.append(locator)
            evidence[locator] = {
                "kind": "text_block",
                "quote": quote,
                "grounding_text": "\n".join([document_scope, *section_path, quote]),
                "section_path": section_path,
                "line_start": start + 1,
                "line_end": index,
            }
            rendered.append(f"[BLOCK {locator}] {quote}")
            last_node_kind = "block"
            last_table = None
            first_semantic_node = False

    source_issues = (
        ["source microtext is not fully character-verifiable"]
        if pdf_page in source_limited_pages
        else []
    )
    compact_text = "\n".join(rendered)
    return {
        "schema_version": UNIT_SCHEMA_VERSION,
        "unit_id": page_locator,
        "kind": "page",
        "document_id": document_id,
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "locator": page_locator,
        "text": compact_text,
        "context": {
            "locators": locators,
            "evidence": evidence,
            "entry_section_path": entry_section_path,
            "exit_section_path": [value for value in section_stack if value],
            "adapter_version": ADAPTER_VERSION,
        },
        "token_ids": [],
        "source_status": "source_limited" if source_issues else "canonical",
        "source_issues": source_issues,
        "source_hash": sha256_text(page_text),
        "content_hash": sha256_text(compact_text),
    }, (last_table if last_node_kind == "table" else None)


def prepare_units(input_path: Path, output_path: Path) -> list[EvidenceUnit]:
    """`document.md`를 검증하고 self-contained page unit으로 만든다."""
    if input_path.suffix.lower() != ".md":
        raise ValueError(f"Population input must be Markdown: {input_path}")
    manifest = _manual_manifest(input_path)
    document_id = str(manifest.get("document_id") or "hyu-manual")
    scope_match = re.match(r"(\d{4})-(\d)-", document_id)
    document_scope = (
        f"{scope_match.group(1)}학년도 {scope_match.group(2)}학기 서울캠퍼스"
        if scope_match
        else document_id
    )
    validation = manifest.get("validation")
    audit = (
        validation.get("independent_content_audit")
        if isinstance(validation, dict)
        else None
    )
    source_limited_pages = {
        int(value)
        for value in (
            audit.get("source_limited_pages", []) if isinstance(audit, dict) else []
        )
    }
    outputs = manifest.get("outputs")
    page_manifest = outputs.get("pages", []) if isinstance(outputs, dict) else []
    expected = {
        int(item["pdf_page"]): str(item["sha256"])
        for item in page_manifest
        if isinstance(item, dict)
    }
    pages = _split_pages(input_path.read_text(encoding="utf-8"))
    scope = manifest.get("scope")
    expected_count = scope.get("pdf_pages", -1) if isinstance(scope, dict) else -1
    if len(pages) != int(expected_count):
        raise ValueError("document.md page count differs from the manual manifest")

    units: list[EvidenceUnit] = []
    section_stack: list[str] = []
    previous_table: dict[str, object] | None = None
    seen: set[str] = set()
    for page_text in pages:
        unit, previous_table = _page_unit(
            page_text,
            document_id,
            document_scope,
            section_stack,
            source_limited_pages,
            previous_table,
        )
        page_number = int(unit["pdf_page"] or 0)
        if expected.get(page_number) != sha256_text(page_text):
            raise ValueError(f"Page {page_number} does not match the manual manifest")
        if unit["unit_id"] in seen:
            raise ValueError(f"Duplicate page unit ID: {unit['unit_id']}")
        seen.add(unit["unit_id"])
        units.append(unit)
    if sorted(int(item["pdf_page"] or 0) for item in units) != list(
        range(1, len(units) + 1)
    ):
        raise ValueError("Manual page sequence is incomplete")
    write_jsonl(output_path, units)
    return units
