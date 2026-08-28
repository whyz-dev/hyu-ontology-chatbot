from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from canonical import SCHEMA_VERSION, build_canonical_page
from html_notice import parse_notice
from layout import detect_layout
from markdown_view import render_canonical_markdown
from ocr import extract_page_image, run_ocr
from semantic import link_table_continuations, parse_tables
from validation import build_reference_term_counts, validate_candidate


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
CACHE_ROOT = DATA_ROOT / "cache"
PROCESSED_ROOT = DATA_ROOT / "processed"
MODEL_ROOT = DATA_ROOT / "models" / "paddlex"
NOTICE_URL = "https://policy.hanyang.ac.kr/front/communication/notice/notice-view?id=2341"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pages(value: str) -> list[int]:
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR-grounded parser for the 2026-2 HYU registration book"
    )
    parser.add_argument(
        "--pages",
        default="1-75",
        help="Printed book pages, for example: 2-4,40-41",
    )
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = RAW_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Web source metadata is missing. Run: uv run python data-parser/crawler")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdf_path = RAW_ROOT / manifest["pdf"]["path"]
    if not pdf_path.exists() or _sha256(pdf_path) != manifest["pdf"]["sha256"]:
        raise SystemExit("Source PDF is missing or differs from data/raw/manifest.json")

    notice_path = RAW_ROOT / "web" / "notice" / "notice-2341.html"
    notice = parse_notice(notice_path, NOTICE_URL, _sha256(notice_path))
    notice_tables = notice.pop("tables")
    _write_json(CACHE_ROOT / "notice.json", notice)
    _write_jsonl(CACHE_ROOT / "notice-tables.jsonl", notice_tables)

    ocr_pages: list[dict[str, object]] = []
    canonical_pages: list[dict[str, object]] = []
    facts: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    model_digest: str | None = None
    page_manifest = {
        page["printed_page"]: page
        for page in manifest["pages"]
        if page["printed_page"] is not None
    }
    reference_term_counts = build_reference_term_counts(
        [
            (RAW_ROOT / page["text_path"]).read_text(encoding="utf-8")
            for page in manifest["pages"]
        ]
    )
    previous_candidate: dict[str, object] | None = None

    for printed_page in _pages(args.pages):
        if printed_page not in page_manifest:
            raise SystemExit(f"Printed page is outside this book: {printed_page}")
        page = page_manifest[printed_page]
        pdf_page = int(page["viewer_page"])
        viewer_text = (RAW_ROOT / page["text_path"]).read_text(encoding="utf-8")
        image_path = extract_page_image(
            pdf_path,
            pdf_page,
            CACHE_ROOT / "page-images" / f"page-{pdf_page:03d}.jpg",
        )
        print(f"[page {printed_page}] OCR", flush=True)
        ocr_record = run_ocr(
            image_path,
            pdf_page,
            printed_page,
            viewer_text,
            CACHE_ROOT / "ocr" / f"page-{pdf_page:03d}.json",
            MODEL_ROOT,
            args.force,
        )
        print(f"[page {printed_page}] table layout", flush=True)
        layout_record = detect_layout(
            image_path,
            ocr_record,
            CACHE_ROOT / "layout" / f"page-{pdf_page:03d}.json",
            args.force,
        )
        print(f"[page {printed_page}] Qwen semantic parsing", flush=True)
        candidate = parse_tables(
            ocr_record,
            layout_record,
            args.model,
            CACHE_ROOT / "candidates" / f"page-{pdf_page:03d}.json",
            args.force,
        )
        model_digest = model_digest or candidate["generation"]["model_digest"]
        if previous_candidate is not None:
            link_table_continuations(previous_candidate, candidate)
        print(f"[page {printed_page}] deterministic evidence validation", flush=True)
        validated, page_facts = validate_candidate(
            candidate,
            ocr_record,
            layout_record,
            evidence_ocr_records=[*ocr_pages, ocr_record],
            reference_term_counts=reference_term_counts,
        )

        ocr_pages.append(ocr_record)
        canonical_pages.append(
            build_canonical_page(ocr_record, layout_record, validated)
        )
        facts.extend(page_facts)
        for table in validated.get("tables", []):
            for row_number, row in enumerate(table.get("rows", []), start=1):
                table_issues = [
                    f"table: {issue}"
                    for issue in table["validation"].get("issues", [])
                ]
                combined_issues = [*table_issues, *row["validation"]["issues"]]
                if not combined_issues:
                    continue
                review_rows.append(
                    {
                        "document_id": validated["document_id"],
                        "pdf_page": validated["pdf_page"],
                        "printed_page": validated["printed_page"],
                        "layout_table_id": table["layout_table_id"],
                        "table_title": table.get("title", ""),
                        "row_number": row_number,
                        "values": {
                            cell["column_key"]: cell["value"]
                            for cell in row["cells"]
                        },
                        "issues": combined_issues,
                        "cells": row["cells"],
                    }
                )
        previous_candidate = candidate

    output_paths = {
        "document": PROCESSED_ROOT / "document.jsonl",
        "review": PROCESSED_ROOT / "review.jsonl",
        "preview": PROCESSED_ROOT / "preview.md",
    }
    _write_jsonl(output_paths["document"], canonical_pages)
    _write_jsonl(output_paths["review"], review_rows)
    output_paths["preview"].write_text(
        render_canonical_markdown(canonical_pages), encoding="utf-8"
    )
    _write_json(
        CACHE_ROOT / "run-manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "source_pdf": manifest["pdf"],
            "printed_pages": _pages(args.pages),
            "pdf_pages": [record["pdf_page"] for record in ocr_pages],
            "model": args.model,
            "model_digest": model_digest,
            "counts": {
                "pages": len(ocr_pages),
                "blocks": sum(len(page["blocks"]) for page in canonical_pages),
                "tables": sum(len(page["tables"]) for page in canonical_pages),
                "cells": sum(
                    len(table["raw_cells"])
                    for page in canonical_pages
                    for table in page["tables"]
                ),
                "rows": sum(
                    len(table["normalized_rows"])
                    for page in canonical_pages
                    for table in page["tables"]
                ),
                "auto_accepted_rows": len(facts),
                "uncertain_rows": len(review_rows),
                "structure_review_cells": sum(
                    cell["structure_status"] != "valid"
                    for page in canonical_pages
                    for table in page["tables"]
                    for cell in table["raw_cells"]
                ),
                "semantic_review_tables": sum(
                    table["semantic_annotation"]["status"] != "generated"
                    for page in canonical_pages
                    for table in page["tables"]
                ),
            },
            "outputs": {
                name: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
                for name, path in output_paths.items()
            },
        },
    )
    print(
        f"Completed {len(ocr_pages)} page(s): {len(facts)} auto-accepted rows, "
        f"{len(review_rows)} rows need review",
        flush=True,
    )


if __name__ == "__main__":
    main()
