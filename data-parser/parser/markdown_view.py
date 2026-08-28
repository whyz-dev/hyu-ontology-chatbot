from __future__ import annotations

import re


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _normalized(value: object) -> str:
    return re.sub(r"\s+", "", str(value))


def _render_table(
    lines: list[str],
    table: dict[str, object],
    previous_heading: str | None,
) -> None:
    if _normalized(previous_heading or "") != _normalized(table["title"]):
        lines.extend([f"### {_escape(table['title'] or '제목 없는 표')}", ""])
    if table.get("continuation_of"):
        parent = table["continuation_of"]
        lines.extend(
            [
                f"> 이전 표에서 계속: PDF {parent['pdf_page']} / {parent['layout_table_id']}",
                "",
            ]
        )

    columns = sorted(table["columns"], key=lambda item: item["base_column"])
    headers = [" / ".join(column["header_path"]) for column in columns]
    lines.append("| " + " | ".join(_escape(header) for header in headers) + " | 상태 |")
    lines.append("| " + " | ".join("---" for _ in headers) + " | --- |")
    for row in table["normalized_rows"]:
        values = []
        for column in columns:
            value = row["values"][column["column_id"]]
            rendered = _escape(value["raw_value"] or "—")
            if value["validation_status"] != "auto_accepted":
                rendered = f"⚠️ {rendered}"
            values.append(rendered)
        status = "✅ 자동 승인" if row["validation_status"] == "auto_accepted" else "⚠️ 검수 필요"
        lines.append("| " + " | ".join(values) + f" | {status} |")
    lines.append("")


def render_canonical_markdown(pages: list[dict[str, object]]) -> str:
    lines = [
        "# 2026-2 한양대학교 학사안내 OCR 재구성",
        "",
        "> 이 문서는 canonical JSON의 표 구조를 검수하기 위한 보기입니다. "
        "⚠️ 표시는 자동 승인되지 않은 셀 또는 행입니다.",
        "",
    ]

    for page in pages:
        lines.extend(
            [
                f"## 인쇄 페이지 {page['printed_page']} (PDF {page['pdf_page']})",
                "",
            ]
        )
        items = [
            (block["bbox"][1], block["bbox"][0], "block", block)
            for block in page["blocks"]
            if block["type"] != "footer"
        ]
        items.extend(
            (table["bbox"][1], table["bbox"][0], "table", table)
            for table in page["tables"]
        )
        previous_heading: str | None = None
        for _, _, item_type, item in sorted(items):
            if item_type == "table":
                _render_table(lines, item, previous_heading)
                previous_heading = None
                continue
            block_type = item["type"]
            text = _escape(item["text"])
            if block_type == "heading":
                lines.extend([f"### {text}", ""])
                previous_heading = item["text"]
            elif block_type == "note":
                lines.extend([f"> **주석:** {text}", ""])
                previous_heading = None
            elif block_type == "list_item":
                lines.extend([f"- {text}", ""])
                previous_heading = None
            else:
                lines.extend([text, ""])
                previous_heading = None

    return "\n".join(lines).rstrip() + "\n"
