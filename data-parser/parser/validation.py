from __future__ import annotations

import copy
import re
from collections import Counter
from difflib import SequenceMatcher

from ocr import normalize_text


def build_reference_term_counts(texts: list[str]) -> dict[str, int]:
    return dict(
        Counter(
            term
            for text in texts
            for term in re.findall(r"[가-힣A-Za-z0-9·()_-]+", text)
            if len(term) >= 2
        )
    )


def _possible_entity_variant(
    value: str, reference_term_counts: dict[str, int]
) -> str | None:
    value = value.strip()
    suffixes = ("대학", "학과", "학부", "전공")
    suffix = next((item for item in suffixes if value.endswith(item)), None)
    if suffix is None or " " in value or len(value) < 4:
        return None
    value_count = reference_term_counts.get(value, 0)
    alternatives = [
        (count, term)
        for term, count in reference_term_counts.items()
        if term != value
        and term.endswith(suffix)
        and len(term) == len(value)
        and count >= max(5, value_count * 3)
        and SequenceMatcher(None, value, term).ratio() >= 0.8
    ]
    return max(alternatives, default=(0, None))[1]


def _ordered_text(token_ids: list[str], token_map: dict[str, dict[str, object]]) -> str:
    tokens = [token_map[token_id] for token_id in token_ids if token_id in token_map]
    tokens.sort(key=lambda token: (token["bbox"][1], token["bbox"][0]))
    return " ".join(str(token["text"]) for token in tokens)


def _validate_text(value: str, evidence_text: str) -> tuple[float, bool]:
    left = normalize_text(value)
    right = normalize_text(evidence_text)
    similarity = SequenceMatcher(None, left, right).ratio() if left or right else 1.0
    value_numbers = re.findall(r"\d+(?:[.-]\d+)*", value)
    evidence_numbers = re.findall(r"\d+(?:[.-]\d+)*", evidence_text)
    return similarity, value_numbers == evidence_numbers


def validate_candidate(
    candidate: dict[str, object],
    ocr_record: dict[str, object],
    layout_record: dict[str, object],
    min_ocr_confidence: float = 0.90,
    evidence_ocr_records: list[dict[str, object]] | None = None,
    reference_term_counts: dict[str, int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    validated = copy.deepcopy(candidate)
    evidence_ocr_records = evidence_ocr_records or [ocr_record]
    token_map = {
        token["token_id"]: token
        for record in evidence_ocr_records
        for token in record["tokens"]
    }
    token_layout = layout_record["token_layout"]
    reference_term_counts = reference_term_counts or {}
    verified_rows: list[dict[str, object]] = []

    for table_index, table in enumerate(validated.get("tables", []), start=1):
        column_keys = [column["key"] for column in table.get("columns", [])]
        table_issues: list[str] = []
        if not column_keys or len(column_keys) != len(set(column_keys)):
            table_issues.append("column keys are empty or duplicated")

        layout_table = next(
            (
                item
                for item in layout_record["tables"]
                if item["table_id"] == table.get("layout_table_id")
            ),
            None,
        )
        if layout_table is None:
            table_issues.append("layout table ID does not exist")
            layout_cells: dict[str, dict[str, object]] = {}
        else:
            layout_cells = {cell["cell_id"]: cell for cell in layout_table["cells"]}
            physical_columns = len(layout_table["x_lines"]) - 1
            base_columns = [column.get("base_column") for column in table.get("columns", [])]
            if sorted(base_columns) != list(range(physical_columns)):
                table_issues.append("semantic columns do not cover each physical table column once")
            table_issues.extend(layout_table.get("structure_issues", []))

        title_ids = table.get("title_evidence_token_ids", [])
        title_text = _ordered_text(title_ids, token_map)
        title_similarity, title_numbers_match = _validate_text(table.get("title", ""), title_text)
        if not title_ids or title_similarity < 0.98 or not title_numbers_match:
            table_issues.append("table title is not grounded in OCR tokens")

        expected_columns: dict[str, int] = {}
        entity_column_keys: set[str] = set()
        for column in table.get("columns", []):
            expected_columns[column["key"]] = column.get("base_column")
            if normalize_text(column.get("label", "")) in {"대학", "학과전공"}:
                entity_column_keys.add(column["key"])
            evidence_ids = column.get("evidence_token_ids", [])
            evidence_text = _ordered_text(evidence_ids, token_map)
            similarity, numbers_match = _validate_text(column.get("label", ""), evidence_text)
            if not evidence_ids or similarity < 0.98 or not numbers_match:
                table_issues.append(f"column {column['key']} label is not grounded in OCR tokens")
                continue
            evidence = [token_id for token_id in evidence_ids if token_id in token_layout]
            if not evidence:
                table_issues.append(f"column {column['key']} has no evidence inside the table")
                continue
            target_column = column.get("base_column")
            if any(
                not (
                    token_layout[token_id]["scope_column_start"]
                    <= target_column
                    <= token_layout[token_id]["scope_column_end"]
                )
                for token_id in evidence
            ):
                table_issues.append(
                    f"column {column['key']} cites a header cell that does not span its physical column"
                )
            deepest = max(evidence, key=lambda token_id: token_map[token_id]["bbox"][1])
            if token_layout[deepest]["base_column"] != target_column:
                table_issues.append(f"column {column['key']} header is in another physical column")
            if token_layout[deepest]["base_row"] > table.get("header_end_row", -1):
                table_issues.append(f"column {column['key']} cites a data row as its header")

        for row_index, row in enumerate(table.get("rows", []), start=1):
            anchor_id = row.get("row_anchor_token_id")
            anchor_layout = token_layout.get(anchor_id)
            row_issues: list[str] = []
            anchor_row = row.get("base_row")
            if not isinstance(anchor_row, int):
                row_issues.append("row has no physical base_row")
                anchor_row = None
            if anchor_id is not None and (
                anchor_id not in token_map or anchor_layout is None
            ):
                row_issues.append("row anchor is outside a detected table")

            cell_keys = [cell["column_key"] for cell in row.get("cells", [])]
            if sorted(cell_keys) != sorted(column_keys):
                row_issues.append("row cells do not match table columns")

            values: dict[str, str] = {}
            evidence_by_column: dict[str, list[str]] = {}
            for cell in row.get("cells", []):
                issues: list[str] = []
                source_cell = layout_cells.get(cell.get("source_cell_id"))
                if source_cell is None:
                    issues.append("source physical cell does not exist")
                evidence_ids = cell.get("evidence_token_ids", [])
                unknown = [token_id for token_id in evidence_ids if token_id not in token_map]
                if unknown:
                    issues.append(f"unknown evidence tokens: {unknown}")
                evidence_text = _ordered_text(evidence_ids, token_map)
                similarity, numbers_match = _validate_text(cell.get("value", ""), evidence_text)
                if cell.get("source_mode") == "empty":
                    if cell.get("value") or evidence_ids:
                        issues.append("empty cell contains a value or evidence")
                else:
                    if not evidence_ids:
                        issues.append("non-empty cell has no evidence")
                    if similarity < 0.98:
                        issues.append(f"OCR text similarity is {similarity:.3f}")
                    if not numbers_match:
                        issues.append("numeric content differs from OCR evidence")

                known_tokens = [token_map[token_id] for token_id in evidence_ids if token_id in token_map]
                if any(token["confidence"] < min_ocr_confidence for token in known_tokens):
                    issues.append("OCR confidence is below threshold")
                if any(not token["viewer_text_match"] for token in known_tokens):
                    issues.append("OCR token is absent from viewer text")
                if cell["column_key"] in entity_column_keys:
                    alternative = _possible_entity_variant(
                        cell.get("value", ""), reference_term_counts
                    )
                    if alternative is not None:
                        issues.append(
                            f"possible OCR variant of frequent corpus term: {alternative}"
                        )

                layouts = [token_layout[token_id] for token_id in evidence_ids if token_id in token_layout]
                expected_column = expected_columns.get(cell["column_key"])
                if source_cell is not None and anchor_row is not None and expected_column is not None:
                    coordinate = [anchor_row, expected_column]
                    if coordinate not in source_cell["base_coordinates"]:
                        issues.append("source cell does not cover the row and column")
                    expected_mode = (
                        "empty"
                        if not source_cell["token_ids"]
                        else "direct"
                        if source_cell["row"] == anchor_row
                        else "rowspan_inheritance"
                    )
                    if cell.get("source_mode") != expected_mode:
                        issues.append("source mode differs from physical cell topology")
                    if any(
                        layout.get("cell_id") != source_cell["cell_id"]
                        for layout in layouts
                    ):
                        issues.append("evidence token belongs to another physical cell")

                cell["validation"] = {
                    "status": "verified" if not issues else "needs_review",
                    "evidence_text": evidence_text,
                    "text_similarity": round(similarity, 6),
                    "issues": issues,
                }
                if issues:
                    row_issues.extend(f"{cell['column_key']}: {issue}" for issue in issues)
                values[cell["column_key"]] = cell.get("value", "")
                evidence_by_column[cell["column_key"]] = evidence_ids

            row["validation"] = {
                "status": "verified" if not row_issues else "needs_review",
                "issues": row_issues,
            }
            if not row_issues and not table_issues:
                verified_rows.append(
                    {
                        "fact_id": f"p{candidate['pdf_page']:03d}-t{table_index:02d}-r{row_index:03d}",
                        "document_id": candidate["document_id"],
                        "pdf_page": candidate["pdf_page"],
                        "printed_page": candidate["printed_page"],
                        "table_title": table.get("title", ""),
                        "table_title_source_pdf_page": table.get(
                            "title_source_pdf_page", candidate["pdf_page"]
                        ),
                        "continuation_of": table.get("continuation_of"),
                        "column_labels": {
                            column["key"]: column["label"] for column in table.get("columns", [])
                        },
                        "values": values,
                        "evidence_token_ids": evidence_by_column,
                        "evidence": {
                            column_key: [
                                {
                                    "token_id": token_id,
                                    "bbox": token_map[token_id]["bbox"],
                                    "confidence": token_map[token_id]["confidence"],
                                    "viewer_text_match": token_map[token_id][
                                        "viewer_text_match"
                                    ],
                                }
                                for token_id in token_ids
                            ]
                            for column_key, token_ids in evidence_by_column.items()
                        },
                        "verification": {
                            "status": "auto_accepted",
                            "semantic_keys": table.get(
                                "semantic_annotation", {}
                            ).get("status", "not_generated"),
                            "labels_and_values": "ocr_grounded",
                        },
                    }
                )

        table["validation"] = {
            "status": "verified"
            if not table_issues and all(row["validation"]["status"] == "verified" for row in table.get("rows", []))
            else "needs_review",
            "issues": table_issues,
        }

    return validated, verified_rows
