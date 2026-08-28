from __future__ import annotations

import re
from typing import Iterable


SCHEMA_VERSION = "canonical-document-v1"


def normalize_typed_value(raw_value: str) -> dict[str, object]:
    value = re.sub(r"\s+", " ", raw_value).strip()
    if not value:
        return {"kind": "empty", "value": None}
    credit = re.fullmatch(r"(\d+)\s*학점\s*(이상|이하)?", value)
    if credit:
        comparator = {"이상": ">=", "이하": "<="}.get(credit.group(2), "=")
        return {
            "kind": "integer",
            "value": int(credit.group(1)),
            "unit": "credit",
            "comparator": comparator,
        }
    if value == "미지정":
        return {"kind": "not_designated", "value": None}
    return {"kind": "string", "value": value}


def _ordered_tokens(
    token_ids: Iterable[str], token_map: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    return sorted(
        (token_map[token_id] for token_id in token_ids),
        key=lambda token: (token["bbox"][1], token["bbox"][0]),
    )


def _bbox(tokens: list[dict[str, object]]) -> list[int]:
    return [
        min(token["bbox"][0] for token in tokens),
        min(token["bbox"][1] for token in tokens),
        max(token["bbox"][2] for token in tokens),
        max(token["bbox"][3] for token in tokens),
    ]


def _page_blocks(
    ocr_record: dict[str, object],
    layout_record: dict[str, object],
    validated: dict[str, object],
) -> list[dict[str, object]]:
    token_map = {token["token_id"]: token for token in ocr_record["tokens"]}
    table_token_ids = set(layout_record["token_layout"])
    title_ids = {
        token_id
        for table in validated.get("tables", [])
        for token_id in table.get("title_evidence_token_ids", [])
        if token_id in token_map
    }
    lines: dict[str, list[str]] = {}
    for token in ocr_record["tokens"]:
        if token["token_id"] not in table_token_ids:
            lines.setdefault(token["line_id"], []).append(token["token_id"])

    page_height = ocr_record["image_size"]["height"]
    token_heights = [
        token["bbox"][3] - token["bbox"][1] for token in ocr_record["tokens"]
    ]
    median_height = sorted(token_heights)[len(token_heights) // 2] if token_heights else 0
    raw_blocks = []
    for token_ids in lines.values():
        tokens = _ordered_tokens(token_ids, token_map)
        text = " ".join(token["text"] for token in tokens)
        bounds = _bbox(tokens)
        if bounds[1] >= page_height * 0.90:
            block_type = "footer"
        elif re.match(r"^\s*(?:※|주\)|주:|\*)", text):
            block_type = "note"
        elif any(token["token_id"] in title_ids for token in tokens):
            block_type = "heading"
        elif re.match(r"^\s*(?:[■▣◆]|\d+[.)]|[가-힣][.)])", text):
            block_type = "list_item"
        elif len(text) <= 60 and bounds[3] - bounds[1] >= median_height * 1.2:
            block_type = "heading"
        else:
            block_type = "paragraph"
        raw_blocks.append(
            {
                "type": block_type,
                "text": text,
                "bbox": bounds,
                "token_ids": [token["token_id"] for token in tokens],
            }
        )

    raw_blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    merged_blocks: list[dict[str, object]] = []
    for block in raw_blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue
        previous = merged_blocks[-1]
        vertical_gap = block["bbox"][1] - previous["bbox"][3]
        compatible = (
            previous["type"] == block["type"] == "paragraph"
            or previous["type"] in {"note", "list_item"}
            and block["type"] == "paragraph"
        )
        aligned = abs(block["bbox"][0] - previous["bbox"][0]) <= 60
        if compatible and aligned and -3 <= vertical_gap <= 14:
            previous["text"] = f"{previous['text']} {block['text']}"
            previous["bbox"] = [
                min(previous["bbox"][0], block["bbox"][0]),
                min(previous["bbox"][1], block["bbox"][1]),
                max(previous["bbox"][2], block["bbox"][2]),
                max(previous["bbox"][3], block["bbox"][3]),
            ]
            previous["token_ids"].extend(block["token_ids"])
        else:
            merged_blocks.append(block)
    return [
        {
            "block_id": f"p{ocr_record['pdf_page']:03d}-block-{index:04d}",
            "reading_order": index,
            **block,
        }
        for index, block in enumerate(merged_blocks, start=1)
    ]


def _table_context(
    table_bbox: list[int], blocks: list[dict[str, object]]
) -> tuple[list[str], list[str]]:
    before = [
        block
        for block in blocks
        if block["type"] != "footer"
        and block["bbox"][3] <= table_bbox[1]
        and table_bbox[1] - block["bbox"][3] <= 200
    ]
    after = [
        block
        for block in blocks
        if block["type"] in {"note", "paragraph"}
        and block["bbox"][1] >= table_bbox[3]
        and block["bbox"][1] - table_bbox[3] <= 300
    ]
    return (
        [block["block_id"] for block in before[-3:]],
        [block["block_id"] for block in after],
    )


def build_canonical_page(
    ocr_record: dict[str, object],
    layout_record: dict[str, object],
    validated: dict[str, object],
) -> dict[str, object]:
    token_map = {token["token_id"]: token for token in ocr_record["tokens"]}
    blocks = _page_blocks(ocr_record, layout_record, validated)
    validated_tables = {
        table["layout_table_id"]: table for table in validated.get("tables", [])
    }
    tables = []

    for layout_table in layout_record["tables"]:
        table = validated_tables[layout_table["table_id"]]
        context_before, context_after = _table_context(layout_table["bbox"], blocks)
        raw_cells = []
        for cell in layout_table["cells"]:
            tokens = _ordered_tokens(cell["token_ids"], token_map)
            raw_text = " ".join(token["text"] for token in tokens)
            raw_cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "row": cell["row"],
                    "column": cell["column"],
                    "rowspan": cell["rowspan"],
                    "colspan": cell["colspan"],
                    "bbox": cell["bbox"],
                    "raw": {
                        "text": raw_text,
                        "token_ids": cell["token_ids"],
                        "ocr_confidence_min": (
                            min(token["confidence"] for token in tokens)
                            if tokens
                            else None
                        ),
                        "viewer_text_match": (
                            all(token["viewer_text_match"] for token in tokens)
                            if tokens
                            else None
                        ),
                    },
                    "structure_status": cell["structure_status"],
                }
            )

        columns = []
        columns_by_key = {}
        for column in sorted(table["columns"], key=lambda item: item["base_column"]):
            column_id = f"{table['layout_table_id']}-column-{column['base_column']}"
            canonical_column = {
                "column_id": column_id,
                "base_column": column["base_column"],
                "header_path": [part.strip() for part in column["label"].split(">")],
                "header_evidence_token_ids": column["evidence_token_ids"],
            }
            columns.append(canonical_column)
            columns_by_key[column["key"]] = canonical_column

        normalized_rows = []
        for row_number, row in enumerate(table.get("rows", []), start=1):
            values = {}
            for cell in row["cells"]:
                column = columns_by_key[cell["column_key"]]
                values[column["column_id"]] = {
                    "raw_value": cell["value"],
                    "normalized_value": normalize_typed_value(cell["value"]),
                    "source_cell_id": cell["source_cell_id"],
                    "inherited_from": (
                        cell["source_cell_id"]
                        if cell["source_mode"] == "rowspan_inheritance"
                        else None
                    ),
                    "evidence_token_ids": cell["evidence_token_ids"],
                    "validation_status": (
                        "auto_accepted"
                        if cell["validation"]["status"] == "verified"
                        else "uncertain"
                    ),
                    "validation_issues": cell["validation"]["issues"],
                }
            effective_issues = [
                *(
                    f"table: {issue}"
                    for issue in table["validation"].get("issues", [])
                ),
                *row["validation"]["issues"],
            ]
            normalized_rows.append(
                {
                    "row_id": f"{table['layout_table_id']}-row-{row_number:04d}",
                    "base_row": row["base_row"],
                    "values": values,
                    "validation_status": (
                        "auto_accepted"
                        if not effective_issues
                        else "uncertain"
                    ),
                    "validation_issues": effective_issues,
                }
            )

        semantic_annotation = table.get(
            "semantic_annotation",
            {"status": "not_generated", "issues": [], "columns": []},
        )
        tables.append(
            {
                "table_id": table["layout_table_id"],
                "bbox": layout_table["bbox"],
                "title": table["title"],
                "title_evidence_token_ids": table["title_evidence_token_ids"],
                "continuation_of": table.get("continuation_of"),
                "context_before_block_ids": context_before,
                "context_after_block_ids": context_after,
                "columns": columns,
                "raw_cells": raw_cells,
                "normalized_rows": normalized_rows,
                "semantic_annotation": {
                    "status": semantic_annotation["status"],
                    "issues": semantic_annotation.get("issues", []),
                    "columns": [
                        {
                            "column_id": next(
                                item["column_id"]
                                for item in columns
                                if item["base_column"] == suggestion["base_column"]
                            ),
                            "suggested_key": suggestion["suggested_key"],
                        }
                        for suggestion in semantic_annotation.get("columns", [])
                    ],
                },
                "validation": table["validation"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": ocr_record["document_id"],
        "pdf_page": ocr_record["pdf_page"],
        "printed_page": ocr_record["printed_page"],
        "image_size": ocr_record["image_size"],
        "source": ocr_record["source"],
        "blocks": blocks,
        "tables": tables,
        "extraction": {
            "ocr": ocr_record["ocr"],
            "layout": layout_record["detector"],
            "semantic_generation": validated["generation"],
        },
    }
