from __future__ import annotations

import copy
import hashlib
import json
import re
from json import JSONDecodeError
from pathlib import Path
from urllib.request import Request, urlopen

from ocr import normalize_text

PROMPT_VERSION = "grounded-header-semantic-key-v4"


def _semantic_id(table_id: str, base_column: int) -> str:
    return f"{table_id.replace('-', '_')}_column_{base_column}"


def _semantic_schema(column_ids: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "semantic_keys": {
                "type": "object",
                "properties": {
                    column_id: {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9_]*$",
                    }
                    for column_id in column_ids
                },
                "required": column_ids,
            }
        },
        "required": ["semantic_keys"],
    }


def _model_digest(model: str) -> str:
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        tags = json.load(response)
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return str(item["digest"])
    raise RuntimeError(f"Ollama model is not installed: {model}")


def _reading_order(token_ids: list[str], token_map: dict[str, dict[str, object]]) -> list[str]:
    return sorted(
        token_ids,
        key=lambda token_id: (
            token_map[token_id]["bbox"][1],
            token_map[token_id]["bbox"][0],
        ),
    )


def _grid_view(
    tokens: list[dict[str, object]],
    layout_record: dict[str, object],
) -> dict[str, object]:
    token_map = {token["token_id"]: token for token in tokens}
    token_layout = layout_record["token_layout"]
    tables = []
    for table in layout_record["tables"]:
        table_id = table["table_id"]
        rows: dict[int, dict[int, list[str]]] = {}
        for token_id in table["token_ids"]:
            layout = token_layout[token_id]
            rows.setdefault(layout["base_row"], {}).setdefault(
                layout["base_column"], []
            ).append(token_id)
        grid_rows = []
        for row_index in sorted(rows):
            cells = []
            for column_index in sorted(rows[row_index]):
                evidence = _reading_order(rows[row_index][column_index], token_map)
                cells.append(
                    {
                        "column": column_index,
                        "text": " ".join(token_map[token_id]["text"] for token_id in evidence),
                        "token_ids": evidence,
                        "token_column_scopes": {
                            token_id: [
                                token_layout[token_id]["scope_column_start"],
                                token_layout[token_id]["scope_column_end"],
                            ]
                            for token_id in evidence
                        },
                    }
                )
            grid_rows.append({"row": row_index, "cells": cells})

        column_count = len(table["x_lines"]) - 1
        dense_threshold = max(2, column_count - 1)
        dense_rows = [
            row["row"] for row in grid_rows if len(row["cells"]) >= dense_threshold
        ]
        if not dense_rows:
            header_end_row = 0
        else:
            data_start_row = dense_rows[0]
            header_end_row = max(0, data_start_row - 1)

        column_scores: list[tuple[float, int]] = []
        for column in range(column_count):
            column_tokens = [
                token_id
                for token_id, item in token_layout.items()
                if item["table_id"] == table_id
                and item["base_column"] == column
                and item["base_row"] > header_end_row
            ]
            occupied_rows = {
                token_layout[token_id]["base_row"] for token_id in column_tokens
            }
            if not column_tokens:
                column_scores.append((0.0, column))
                continue
            nonnumeric_ratio = sum(
                not any(character.isdigit() for character in token_map[token_id]["text"])
                for token_id in column_tokens
            ) / len(column_tokens)
            column_scores.append((len(occupied_rows) * nonnumeric_ratio, column))
        row_anchor_column = max(column_scores)[1]

        tables.append(
            {
                "layout_table_id": table_id,
                "column_count": column_count,
                "row_count": len(table["y_lines"]) - 1,
                "detected_header_end_row": header_end_row,
                "detected_row_anchor_column": row_anchor_column,
                "header_rows": [row for row in grid_rows if row["row"] <= header_end_row],
                "sample_data_rows": [
                    row for row in grid_rows if header_end_row < row["row"] <= header_end_row + 2
                ],
            }
        )
    table_tops = [table["bbox"][1] for table in layout_record["tables"]]
    first_table_top = min(table_tops) if table_tops else 0
    outside_tokens = [
        {
            "token_id": token["token_id"],
            "text": token["text"],
            "bbox": token["bbox"],
        }
        for token in tokens
        if token["token_id"] not in token_layout
        and first_table_top - 100 <= token["bbox"][1] < first_table_top
    ]
    return {"outside_tokens": outside_tokens, "tables": tables}


def _grounded_tables(
    source: dict[str, object],
    tokens: list[dict[str, object]],
    layout_record: dict[str, object],
) -> dict[str, object]:
    token_map = {token["token_id"]: token for token in tokens}
    token_layout = layout_record["token_layout"]
    source_tables = {table["layout_table_id"]: table for table in source["tables"]}
    title_ids = _reading_order(
        [token["token_id"] for token in source["outside_tokens"]], token_map
    )
    title = " ".join(token_map[token_id]["text"] for token_id in title_ids)
    tables: list[dict[str, object]] = []

    for layout_table in layout_record["tables"]:
        table_id = layout_table["table_id"]
        inferred = source_tables[table_id]
        header_end_row = inferred["detected_header_end_row"]
        columns = []
        for target_column in range(inferred["column_count"]):
            ids_by_row: dict[int, list[str]] = {}
            for token_id in layout_table["token_ids"]:
                item = token_layout[token_id]
                if (
                    item["base_row"] <= header_end_row
                    and item["scope_column_start"]
                    <= target_column
                    <= item["scope_column_end"]
                ):
                    ids_by_row.setdefault(item["base_row"], []).append(token_id)
            evidence_ids: list[str] = []
            label_parts: list[str] = []
            for row_index in sorted(ids_by_row):
                row_ids = _reading_order(ids_by_row[row_index], token_map)
                evidence_ids.extend(row_ids)
                label_parts.append(
                    " ".join(token_map[token_id]["text"] for token_id in row_ids)
                )
            columns.append(
                {
                    "label": " > ".join(label_parts),
                    "base_column": target_column,
                    "evidence_token_ids": evidence_ids,
                }
            )
        tables.append(
            {
                "layout_table_id": table_id,
                "title": title,
                "title_evidence_token_ids": title_ids,
                "header_end_row": header_end_row,
                "row_anchor_column": inferred["detected_row_anchor_column"],
                "columns": columns,
            }
        )
    return {"tables": tables}


def _materialize_rows(
    parsed: dict[str, object],
    tokens: list[dict[str, object]],
    layout_record: dict[str, object],
) -> None:
    token_map = {token["token_id"]: token for token in tokens}
    token_layout = layout_record["token_layout"]
    layout_tables = {table["table_id"]: table for table in layout_record["tables"]}
    for table in parsed.get("tables", []):
        table_id = table["layout_table_id"]
        header_end_row = table["header_end_row"]
        anchor_column = table["row_anchor_column"]
        layout_table = layout_tables[table_id]
        cells_by_coordinate = {
            tuple(coordinate): cell
            for cell in layout_table["cells"]
            for coordinate in cell["base_coordinates"]
        }
        data_rows = sorted(
            {
                layout["base_row"]
                for layout in token_layout.values()
                if layout["table_id"] == table_id
                and layout["base_row"] > header_end_row
            }
        )

        rows = []
        for anchor_row in data_rows:
            anchor_ids = [
                token_id
                for token_id, layout in token_layout.items()
                if layout["table_id"] == table_id
                and layout["base_column"] == anchor_column
                and layout["base_row"] == anchor_row
            ]
            if not anchor_ids:
                anchor_ids = [
                    token_id
                    for token_id, layout in token_layout.items()
                    if layout["table_id"] == table_id
                    and layout["base_row"] == anchor_row
                ]
            ordered_anchor_ids = _reading_order(anchor_ids, token_map)
            anchor_id = ordered_anchor_ids[0] if ordered_anchor_ids else None
            cells = []
            for column in sorted(table["columns"], key=lambda item: item["base_column"]):
                column_key = column["key"]
                expected_column = column["base_column"]
                source_cell = cells_by_coordinate[(anchor_row, expected_column)]
                evidence = _reading_order(source_cell["token_ids"], token_map)
                if not evidence:
                    mode = "empty"
                elif source_cell["row"] == anchor_row:
                    mode = "direct"
                else:
                    mode = "rowspan_inheritance"
                cells.append(
                    {
                        "column_key": column_key,
                        "value": " ".join(token_map[token_id]["text"] for token_id in evidence),
                        "evidence_token_ids": evidence,
                        "source_cell_id": source_cell["cell_id"],
                        "source_mode": mode,
                    }
                )
            rows.append(
                {
                    "base_row": anchor_row,
                    "row_anchor_token_id": anchor_id,
                    "cells": cells,
                }
            )
        table["rows"] = rows


def _semantic_issues(
    parsed: dict[str, object],
    grounded: dict[str, object],
) -> list[str]:
    issues: list[str] = []
    expected_ids = {
        _semantic_id(table["layout_table_id"], column["base_column"])
        for table in grounded["tables"]
        for column in table["columns"]
    }
    semantic_keys = parsed.get("semantic_keys", {})
    if not isinstance(semantic_keys, dict):
        return ["semantic_keys가 JSON object가 아닙니다"]
    if set(semantic_keys) != expected_ids:
        issues.append(
            f"semantic column ID가 다릅니다. 누락={sorted(expected_ids - set(semantic_keys))}, "
            f"초과={sorted(set(semantic_keys) - expected_ids)}"
        )
    for table in grounded["tables"]:
        keys = [
            semantic_keys.get(_semantic_id(table["layout_table_id"], column["base_column"]))
            for column in table["columns"]
        ]
        if len(keys) != len(set(keys)):
            issues.append(f"{table['layout_table_id']}: semantic key가 중복되었습니다")
        for key in keys:
            if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
                issues.append(f"{table['layout_table_id']}: 잘못된 semantic key={key!r}")
    return issues


def _attach_semantic_keys(
    grounded: dict[str, object],
    parsed: dict[str, object],
    semantic_status: str,
    semantic_issues: list[str],
) -> dict[str, object]:
    result = copy.deepcopy(grounded)
    semantic_keys = parsed.get("semantic_keys", {})
    for table in result["tables"]:
        suggestions = []
        for column in table["columns"]:
            semantic_id = _semantic_id(
                table["layout_table_id"], column["base_column"]
            )
            column["key"] = f"column_{column['base_column']}"
            column["key_source"] = "physical_column_id"
            if semantic_id in semantic_keys:
                suggestions.append(
                    {
                        "base_column": column["base_column"],
                        "suggested_key": semantic_keys[semantic_id],
                    }
                )
        table["semantic_annotation"] = {
            "status": semantic_status,
            "issues": semantic_issues,
            "columns": suggestions,
        }
    return result


def link_table_continuations(
    previous: dict[str, object], current: dict[str, object]
) -> None:
    if current["pdf_page"] != previous["pdf_page"] + 1:
        return

    previous_by_signature: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for table in previous.get("tables", []):
        signature = tuple(
            normalize_text(column["label"])
            for column in sorted(table["columns"], key=lambda item: item["base_column"])
        )
        previous_by_signature.setdefault(signature, []).append(table)

    for table in current.get("tables", []):
        if table.get("title") or table.get("title_evidence_token_ids"):
            continue
        signature = tuple(
            normalize_text(column["label"])
            for column in sorted(table["columns"], key=lambda item: item["base_column"])
        )
        matches = previous_by_signature.get(signature, [])
        if len(matches) != 1:
            continue
        parent = matches[0]
        parent_columns = {
            normalize_text(column["label"]): column for column in parent["columns"]
        }
        for column in table["columns"]:
            parent_column = parent_columns[normalize_text(column["label"])]
            column["key"] = parent_column["key"]
            column["key_source"] = "physical_column_id"
        if parent.get("semantic_annotation", {}).get("status") == "generated":
            table["semantic_annotation"] = copy.deepcopy(parent["semantic_annotation"])
            table["semantic_annotation"]["source"] = "continued_table_schema"
        table["title"] = parent["title"]
        table["title_evidence_token_ids"] = parent["title_evidence_token_ids"]
        table["title_source_pdf_page"] = parent.get(
            "title_source_pdf_page", previous["pdf_page"]
        )
        table["continuation_of"] = {
            "pdf_page": previous["pdf_page"],
            "layout_table_id": parent["layout_table_id"],
        }


def parse_tables(
    ocr_record: dict[str, object],
    layout_record: dict[str, object],
    model: str,
    cache_path: Path,
    force: bool = False,
) -> dict[str, object]:
    source = _grid_view(ocr_record["tokens"], layout_record)
    grounded = _grounded_tables(source, ocr_record["tokens"], layout_record)
    semantic_input = {
        "tables": [
            {
                "layout_table_id": table["layout_table_id"],
                "title": table["title"],
                "columns": [
                    {
                        "semantic_id": _semantic_id(
                            table["layout_table_id"], column["base_column"]
                        ),
                        "base_column": column["base_column"],
                        "label": column["label"],
                    }
                    for column in table["columns"]
                ],
            }
            for table in grounded["tables"]
        ]
    }
    source_json = json.dumps(semantic_input, ensure_ascii=False, separators=(",", ":"))
    column_ids = [
        column["semantic_id"]
        for table in semantic_input["tables"]
        for column in table["columns"]
    ]
    source_sha256 = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        generation = cached.get("generation", {})
        if (
            generation.get("source_sha256") == source_sha256
            and generation.get("prompt_version") == PROMPT_VERSION
            and generation.get("model") == model
        ):
            return cached

    prompt = """좌표 분석으로 확정된 한국어 표 제목과 열 label에 영어 semantic key만 부여하세요.
semantic_keys object에 입력의 semantic_id를 모두 정확히 한 번씩 key로 넣으세요.
각 값은 해당 label의 전체 의미를 표현하는 고유한 영어 snake_case여야 합니다.
title이나 label이나 표의 값은 출력하지 마세요.
외부 지식으로 사실을 추가하거나 열을 합치거나 나누지 마세요.

[GROUNDED HEADER JSON]
""" + source_json
    digest = _model_digest(model)
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": _semantic_schema(column_ids),
        "options": {"temperature": 0, "seed": 42, "num_predict": 1000},
        "messages": [{"role": "user", "content": prompt}],
    }
    def request_parse(messages: list[dict[str, str]]) -> tuple[dict[str, object], str]:
        payload["messages"] = messages
        request = Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=1200) as response:
            ollama_result = json.load(response)
        content = ollama_result["message"]["content"]
        try:
            return json.loads(content), content
        except JSONDecodeError as error:
            invalid_path = cache_path.with_suffix(".invalid.txt")
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_path.write_text(content, encoding="utf-8")
            raise RuntimeError(f"Qwen returned invalid JSON; preserved at {invalid_path}") from error

    messages = [{"role": "user", "content": prompt}]
    parsed: dict[str, object] = {"semantic_keys": {}}
    semantic_issues: list[str] = []
    semantic_status = "generated"
    if column_ids:
        try:
            parsed, content = request_parse(messages)
            semantic_issues = _semantic_issues(parsed, grounded)
            if semantic_issues:
                correction = (
                    "이전 응답은 다음 결정적 검사를 통과하지 못했습니다:\n- "
                    + "\n- ".join(semantic_issues)
                    + "\nGROUNDED HEADER JSON만 사용하여 전체 JSON을 수정해서 다시 반환하세요."
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": correction},
                    ]
                )
                parsed, _ = request_parse(messages)
                semantic_issues = _semantic_issues(parsed, grounded)
        except RuntimeError as error:
            semantic_issues = [str(error)]
        if semantic_issues:
            semantic_status = "needs_review"
            rejected_path = cache_path.with_suffix(".rejected.json")
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_path.write_text(
                json.dumps(
                    {"candidate": parsed, "issues": semantic_issues},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            cache_path.with_suffix(".invalid.txt").unlink(missing_ok=True)
            cache_path.with_suffix(".rejected.json").unlink(missing_ok=True)
    parsed = _attach_semantic_keys(
        grounded, parsed, semantic_status, semantic_issues
    )
    _materialize_rows(parsed, ocr_record["tokens"], layout_record)

    record = {
        "document_id": ocr_record["document_id"],
        "pdf_page": ocr_record["pdf_page"],
        "printed_page": ocr_record["printed_page"],
        **parsed,
        "generation": {
            "model": model,
            "model_digest": digest,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source_sha256": source_sha256,
            "semantic_status": semantic_status,
            "semantic_issues": semantic_issues,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
