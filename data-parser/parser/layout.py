from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


LAYOUT_VERSION = "opencv-line-grid-v2"


def _clusters(indices: np.ndarray) -> list[int]:
    if not len(indices):
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value <= groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def _between(value: float, boundaries: list[int]) -> int | None:
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= value <= end:
            return index
    return None


def _containing_scope(value: float, boundaries: list[int]) -> tuple[int, int] | None:
    index = _between(value, boundaries)
    if index is None:
        return None
    return boundaries[index], boundaries[index + 1]


def _has_vertical_boundary(
    vertical: np.ndarray,
    x_line: int,
    y_start: int,
    y_end: int,
    threshold: float = 0.45,
) -> bool:
    start = min(y_end, y_start + 2)
    end = max(start + 1, y_end - 1)
    strip = vertical[start:end, max(0, x_line - 2) : x_line + 3]
    if not strip.size:
        return False
    return float(np.count_nonzero(strip, axis=1).astype(bool).mean()) >= threshold


def _has_horizontal_boundary(
    horizontal: np.ndarray,
    y_line: int,
    x_start: int,
    x_end: int,
    threshold: float = 0.45,
) -> bool:
    start = min(x_end, x_start + 2)
    end = max(start + 1, x_end - 1)
    strip = horizontal[max(0, y_line - 2) : y_line + 3, start:end]
    if not strip.size:
        return False
    return float(np.count_nonzero(strip, axis=0).astype(bool).mean()) >= threshold


def _table_cells(
    table_id: str,
    x_lines: list[int],
    y_lines: list[int],
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], str], list[str]]:
    row_count = len(y_lines) - 1
    column_count = len(x_lines) - 1
    node_count = row_count * column_count
    parents = list(range(node_count))

    def node(row: int, column: int) -> int:
        return row * column_count + column

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for row in range(row_count):
        for column in range(column_count):
            if column + 1 < column_count and not _has_vertical_boundary(
                vertical,
                x_lines[column + 1],
                y_lines[row],
                y_lines[row + 1],
            ):
                union(node(row, column), node(row, column + 1))
            if row + 1 < row_count and not _has_horizontal_boundary(
                horizontal,
                y_lines[row + 1],
                x_lines[column],
                x_lines[column + 1],
            ):
                union(node(row, column), node(row + 1, column))

    components: dict[int, list[tuple[int, int]]] = {}
    for row in range(row_count):
        for column in range(column_count):
            components.setdefault(find(node(row, column)), []).append((row, column))

    ordered = sorted(
        components.values(),
        key=lambda positions: (min(row for row, _ in positions), min(col for _, col in positions)),
    )
    cells: list[dict[str, object]] = []
    coordinate_to_cell: dict[tuple[int, int], str] = {}
    issues: list[str] = []
    for cell_number, positions in enumerate(ordered, start=1):
        row_start = min(row for row, _ in positions)
        row_end = max(row for row, _ in positions)
        column_start = min(column for _, column in positions)
        column_end = max(column for _, column in positions)
        expected_positions = {
            (row, column)
            for row in range(row_start, row_end + 1)
            for column in range(column_start, column_end + 1)
        }
        rectangular = set(positions) == expected_positions
        cell_id = f"{table_id}-cell-{cell_number:04d}"
        if not rectangular:
            issues.append(f"{cell_id}: merged component is not rectangular")
        for position in positions:
            coordinate_to_cell[position] = cell_id
        cells.append(
            {
                "cell_id": cell_id,
                "row": row_start,
                "column": column_start,
                "rowspan": row_end - row_start + 1,
                "colspan": column_end - column_start + 1,
                "bbox": [
                    x_lines[column_start],
                    y_lines[row_start],
                    x_lines[column_end + 1],
                    y_lines[row_end + 1],
                ],
                "base_coordinates": [list(position) for position in sorted(positions)],
                "token_ids": [],
                "structure_status": "valid" if rectangular else "needs_review",
            }
        )
    return cells, coordinate_to_cell, issues


def detect_layout(
    image_path: Path,
    ocr_record: dict[str, object],
    cache_path: Path,
    force: bool = False,
) -> dict[str, object]:
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("detector", {}).get("version") == LAYOUT_VERSION:
            return cached

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = max(20, width // 40)
    vertical_kernel = max(15, height // 80)
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel, 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel)),
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width >= width * 0.35 and box_height >= 60:
            candidates.append((x, y, box_width, box_height))
    candidates.sort(key=lambda box: (box[1], box[0]))

    tables: list[dict[str, object]] = []
    token_layout: dict[str, dict[str, object]] = {}
    for table_number, (x, y, box_width, box_height) in enumerate(candidates, start=1):
        vertical_region = vertical[y : y + box_height + 1, x : x + box_width + 1]
        horizontal_region = horizontal[y : y + box_height + 1, x : x + box_width + 1]
        x_strength = np.count_nonzero(vertical_region, axis=0)
        y_strength = np.count_nonzero(horizontal_region, axis=1)
        x_lines = sorted(
            {
                x,
                x + box_width - 1,
                *(x + value for value in _clusters(np.flatnonzero(x_strength >= box_height * 0.20))),
            }
        )
        y_lines = sorted(
            {
                y,
                y + box_height - 1,
                *(y + value for value in _clusters(np.flatnonzero(y_strength >= box_width * 0.20))),
            }
        )
        if len(x_lines) < 2 or len(y_lines) < 2:
            continue

        table_id = f"p{ocr_record['pdf_page']:03d}-table-{table_number:02d}"
        cells, coordinate_to_cell, structure_issues = _table_cells(
            table_id,
            x_lines,
            y_lines,
            horizontal,
            vertical,
        )
        cells_by_id = {cell["cell_id"]: cell for cell in cells}
        table_token_ids: list[str] = []
        for token in ocr_record["tokens"]:
            x0, y0, x1, y1 = token["bbox"]
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            column = _between(center_x, x_lines)
            row = _between(center_y, y_lines)
            if column is None or row is None:
                continue

            band_start, band_end = x_lines[column], x_lines[column + 1]
            band = horizontal[y : y + box_height + 1, band_start : band_end + 1]
            band_width = max(1, band_end - band_start + 1)
            band_strength = np.count_nonzero(band, axis=1)
            local_y_boundaries = sorted(
                {
                    y,
                    y + box_height - 1,
                    *(
                        y + value
                        for value in _clusters(
                            np.flatnonzero(band_strength >= band_width * 0.45)
                        )
                    ),
                }
            )
            scope = _containing_scope(center_y, local_y_boundaries)
            if scope is None:
                scope = (y_lines[row], y_lines[row + 1])
            scope_start = _between(scope[0] + 1, y_lines)
            scope_end = _between(scope[1] - 1, y_lines)

            row_start, row_end = y_lines[row], y_lines[row + 1]
            row_band = vertical[row_start : row_end + 1, x : x + box_width + 1]
            row_height = max(1, row_end - row_start + 1)
            row_strength = np.count_nonzero(row_band, axis=0)
            local_x_boundaries = sorted(
                {
                    x,
                    x + box_width - 1,
                    *(
                        x + value
                        for value in _clusters(
                            np.flatnonzero(row_strength >= row_height * 0.45)
                        )
                    ),
                }
            )
            column_scope = _containing_scope(center_x, local_x_boundaries)
            if column_scope is None:
                column_scope = (x_lines[column], x_lines[column + 1])
            column_scope_start = _between(column_scope[0] + 1, x_lines)
            column_scope_end = _between(column_scope[1] - 1, x_lines)

            token_id = token["token_id"]
            cell_id = coordinate_to_cell[(row, column)]
            table_token_ids.append(token_id)
            cells_by_id[cell_id]["token_ids"].append(token_id)
            token_layout[token_id] = {
                "table_id": table_id,
                "cell_id": cell_id,
                "base_row": row,
                "base_column": column,
                "scope_row_start": row if scope_start is None else scope_start,
                "scope_row_end": row if scope_end is None else scope_end,
                "scope_column_start": (
                    column if column_scope_start is None else column_scope_start
                ),
                "scope_column_end": column if column_scope_end is None else column_scope_end,
            }

        tables.append(
            {
                "table_id": table_id,
                "bbox": [x, y, x + box_width, y + box_height],
                "x_lines": x_lines,
                "y_lines": y_lines,
                "token_ids": table_token_ids,
                "cells": cells,
                "structure_issues": structure_issues,
            }
        )

    record = {
        "document_id": ocr_record["document_id"],
        "pdf_page": ocr_record["pdf_page"],
        "printed_page": ocr_record["printed_page"],
        "tables": tables,
        "token_layout": token_layout,
        "detector": {
            "name": "opencv-line-grid",
            "version": LAYOUT_VERSION,
            "threshold": 200,
            "horizontal_kernel": horizontal_kernel,
            "vertical_kernel": vertical_kernel,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
