"""원본 페이지 이미지에서 검수 가능한 Markdown 문서를 복원한다.

기존 ``processed/document.jsonl``은 좌표 기반 parser 산출물로 그대로 보존한다.
이 모듈은 Qwen vision에 페이지 이미지 전체와 공식 e-book 텍스트를 함께 제공하고,
표와 문단의 읽기 순서만 Markdown으로 복원한다. 모델이 내용을 요약하거나 의미를
추론하지 못하도록 하고, 생성 후 숫자와 단어를 원본 OCR/e-book 텍스트와 대조한다.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import re
import subprocess
import unicodedata
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
CACHE_ROOT = DATA_ROOT / "cache"
RESTORED_ROOT = DATA_ROOT / "restored"
RESTORATION_IMAGE_ROOT = CACHE_ROOT / "restoration-images"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
SCHEMA_VERSION = "restored-markdown-page-v1"
RESTORATION_VERSION = "vision-markdown-v11"
RENDER_DPI = 250
MAX_REPAIR_ATTEMPTS = 2

NUMBER_PATTERN = re.compile(r"\d+")
WORD_PATTERN = re.compile(r"[A-Za-z가-힣]{2,}")
MARKDOWN_STRUCTURE_WORDS = {"구분"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _model_digest(base_url: str, model: str) -> str:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=10
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Cannot connect to Ollama: {error}") from error
    for item in payload.get("models", []):
        if isinstance(item, dict) and item.get("name") == model:
            return str(item.get("digest", ""))
    raise RuntimeError(f"Ollama model is not installed: {model}")


def _output_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "markdown": {"type": "string"},
            "uncertainties": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["location", "reason"],
                    "additionalProperties": False,
                },
            },
            "audit": {
                "type": "object",
                "properties": {
                    "all_visible_text_transcribed": {"type": "boolean"},
                    "table_rows_aligned": {"type": "boolean"},
                    "merged_cells_preserved": {"type": "boolean"},
                    "no_added_content": {"type": "boolean"},
                    "issues": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "all_visible_text_transcribed",
                    "table_rows_aligned",
                    "merged_cells_preserved",
                    "no_added_content",
                    "issues",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["markdown", "uncertainties", "audit"],
        "additionalProperties": False,
    }


def _geometry_hints(
    table_geometry: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "table_id": table.get("table_id"),
            "detected_min_columns": table.get("physical_columns"),
        }
        for table in table_geometry
    ]


def _prompt(printed_page: int, table_geometry: list[dict[str, object]]) -> str:
    return f"""첨부된 한양대학교 수강편람 이미지 한 페이지를 Markdown으로 정확히 전사한다.
첫 이미지는 전체 페이지이고, 두 번째 이미지는 같은 페이지 중앙의 확대본이다.
확대본을 별도 페이지로 취급하거나 내용을 중복해서 쓰지 않는다.

[목표]
- 이미지의 읽기 순서, 제목, 문단, 목록, 표 구조를 복원한다.
- 요약, 설명, 온톨로지 변환, 의미 추론은 하지 않는다.

[필수 규칙]
1. 이미지에 보이는 정보만 쓴다. 내용을 추가·삭제·교정·의역하지 않는다.
2. 머리말의 장 번호와 페이지 하단 쪽수·학교 로고는 생략한다.
3. 제목은 Markdown heading, 문장은 문단, 항목은 목록으로 표현한다.
4. 표는 Markdown table로 만든다. 이미지의 한 논리 행이 Markdown 한 행이어야 한다.
5. rowspan/병합 셀도 표의 물리 열 수를 유지한다.
   - 상단의 `구분` 헤더 하나가 본문의 여러 세로 칸을 덮으면 칸을 합치지 않는다.
     실제 세로 경계마다 Markdown 열을 만들고 헤더에는 `구분`을 반복한다.
   - 병합 셀의 값은 시작 행·열에 한 번만 쓰고 덮이는 뒤의 행·열은 빈칸으로 둔다.
   - header, separator, 모든 body row의 Markdown 열 개수는 반드시 같아야 한다.
   - 비고 문장은 이미지에서 그 셀이 세로로 걸치는 행에만 붙인다. 셀의 아래 경계를
     넘어 다음 행의 비고를 앞 행에 붙이지 않는다.
   - 하나의 비고 셀이 여러 행에 걸쳐 있으면 비고의 일부 항목을 행별로 나누지 않는다.
     전체 내용을 시작 행에 한 번 쓰고 적용되는 다음 행은 빈칸으로 둔다.
6. 날짜·시간·학점·학년·학수번호·부정·예외를 원문 그대로 보존한다.
7. 표 구조와 글자는 모두 첨부 이미지에서 읽는다. 다른 문서 구조를 가정하지 않는다.
8. 확실히 읽을 수 없는 값은 추측하지 말고 Markdown에 `[불확실]`로 표시하고
   uncertainties에 위치와 이유를 기록한다.
   `[불확실]` 표시가 없는 정상적인 병합 셀·복수 날짜·빈칸은 uncertainties나
   audit issues에 기록하지 않는다.
9. Markdown 코드 펜스는 쓰지 않는다.
10. 출력 전 이미지의 모든 표를 왼쪽에서 오른쪽, 위에서 아래로 한 행씩 다시 따라가며
    각 값과 비고가 같은 가로 행에 놓였는지 audit에 기록한다.
11. 아래 표 힌트는 이미지 선 검출로 얻은 최소 열 수이며 가로선·세로선을 놓칠 수 있다.
    이미지에서 더 많은 열이 일관되게 보이면 더 많은 쪽을 따른다. 모든 Markdown 행은
    최종적으로 판단한 최대 열 수를 유지한다.
12. OCR text의 공백은 토큰 경계일 수 있으므로 그대로 복사하지 않는다. 이미지에서
    붙어 있는 `1학년`, `0시`, `2학기`, `2과목` 같은 표기는 불필요하게 띄우지 않는다.
13. HTML이나 `rowspan`, `colspan` 속성을 출력하지 않는다. Markdown 표만 쓴다.

[인쇄 페이지]
{printed_page}

[검출된 표 격자]
{json.dumps(_geometry_hints(table_geometry), ensure_ascii=False)}
"""


def _verification_prompt(
    printed_page: int,
    table_geometry: list[dict[str, object]],
    draft_markdown: str,
    detected_issues: list[str] | None = None,
) -> str:
    issue_text = "\n".join(f"- {issue}" for issue in (detected_issues or []))
    if not issue_text:
        issue_text = "- 자동 검출 오류 없음. 이미지와 직접 대조할 것."
    return f"""첨부된 이미지는 한 페이지의 전체 이미지와 중앙 확대본이다.
이 원본 페이지 이미지와 아래 Markdown 초안을 줄 단위로 대조해
잘못 전사된 부분만 교정한다. 결과는 교정된 전체 Markdown이어야 한다.

[검증 규칙]
1. 이미지에 없는 내용을 추가하거나 요약·의역하지 않는다.
2. 제목의 오탈자, 빠진 문장, 잘못 붙인 목록을 확인한다.
3. 표는 header, separator, 모든 body row의 열 개수가 같아야 한다. 상단의 `구분`
   헤더가 본문의 여러 세로 칸을 덮으면 실제 세로 칸 수만큼 `구분` 열을 반복한다.
4. 표의 각 행을 이미지의 가로 방향으로 추적해 대분류·세부 구분·대상·날짜·비고가
   같은 행에 있는지 확인한다. 특히 다음 행의 비고를 앞 행에 붙이지 않는다.
5. 병합된 행 헤더도 물리 열을 없애지 않는다. 값은 시작 위치에 한 번 쓰고 덮이는
   행·열은 빈칸으로 둔다. 하나의 비고 셀이 여러 행을 덮으면 비고 항목을 행별로
   분배하지 말고 전체 비고를 시작 행에 한 번만 쓴다.
6. 날짜, 시간, 숫자, 부정과 예외를 이미지와 글자 단위로 다시 확인한다.
7. 확실하지 않은 값은 추측하지 말고 `[불확실]`로 표시한다.
8. Markdown 코드 펜스는 쓰지 않는다.
9. 페이지 하단 인쇄 쪽수와 HANYANG UNIVERSITY 로고는 제거한다.
10. 교정 후 네 가지 audit boolean은 실제 대조 결과에 따라 기록한다. 하나라도
    확신할 수 없으면 false로 두고 issues에 구체적인 위치를 쓴다.
    올바르게 표현된 병합 셀·복수 날짜·빈칸은 issue가 아니다.
11. 아래 표 힌트의 열 수는 최소값이다. 이미지와 초안 중 더 많은 일관된 열 수를 따라
    모든 행을 맞춘다. 병합 셀은 시작 위치에만 두며 덮이는 위치는 빈칸으로 유지한다.
12. HTML이나 `rowspan`, `colspan` 속성을 출력하지 않는다. Markdown 표만 쓴다.

[인쇄 페이지]
{printed_page}

[검출된 표 격자]
{json.dumps(_geometry_hints(table_geometry), ensure_ascii=False)}

[자동 검출된 문제]
{issue_text}

[1차 Markdown 초안]
{draft_markdown}
"""


def _render_pdf_page(pdf_path: Path, pdf_page: int) -> Path:
    """작은 웹 이미지 대신 PDF의 한 페이지만 Qwen 입력용으로 렌더링한다."""

    output_path = RESTORATION_IMAGE_ROOT / f"page-{pdf_page:03d}.jpg"
    if output_path.is_file():
        return output_path
    RESTORATION_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(pdf_page),
                "-l",
                str(pdf_page),
                "-r",
                str(RENDER_DPI),
                "-jpeg",
                "-singlefile",
                str(pdf_path),
                str(output_path.with_suffix("")),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("pdftoppm is required to render PDF pages") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to render PDF page {pdf_page}: {error.stderr.strip()}"
        ) from error
    if not output_path.is_file():
        raise RuntimeError(f"PDF renderer did not create {output_path}")
    return output_path


def _page_images(image_path: Path) -> list[Path]:
    """한 페이지 단위를 유지하면서 중앙의 작은 표 글씨만 확대한다."""

    images = [image_path]
    with Image.open(image_path) as source:
        width, height = source.size
        regions = {"middle": (0, int(height * 0.16), width, int(height * 0.86))}
        for name, box in regions.items():
            crop_path = image_path.with_name(f"{image_path.stem}-{name}.jpg")
            if not crop_path.is_file():
                source.crop(box).save(crop_path, "JPEG", quality=92)
            images.append(crop_path)
    return images


def _clean_markdown(markdown: str) -> str:
    """프롬프트를 어기고 남긴 코드 펜스와 페이지 footer만 제거한다."""

    lines = []
    value = markdown.replace("```markdown", "").replace("```", "")
    for line in value.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", stripped):
            continue
        if stripped.upper() == "HANYANG UNIVERSITY":
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def _table_geometry(
    pdf_page: int, ocr_record: dict[str, object]
) -> list[dict[str, object]]:
    """canonical 값이 아닌 원시 격자와 셀 OCR만 Qwen 보조 근거로 압축한다."""

    layout_path = CACHE_ROOT / "layout" / f"page-{pdf_page:03d}.json"
    if not layout_path.is_file():
        return []
    layout = _read_json(layout_path)
    token_by_id = {
        str(token.get("token_id")): token
        for token in ocr_record.get("tokens", [])
        if isinstance(token, dict)
    }
    tables = []
    for table in layout.get("tables", []):
        if not isinstance(table, dict):
            continue
        cells = []
        for cell in table.get("cells", []):
            if not isinstance(cell, dict):
                continue
            tokens = [
                token_by_id[token_id]
                for token_id in cell.get("token_ids", [])
                if token_id in token_by_id
            ]
            tokens.sort(
                key=lambda token: (
                    str(token.get("line_id", "")),
                    int(token.get("line_word_index", 0)),
                )
            )
            cells.append(
                {
                    "row": int(cell.get("row", 0)),
                    "column": int(cell.get("column", 0)),
                    "rowspan": int(cell.get("rowspan", 1)),
                    "colspan": int(cell.get("colspan", 1)),
                    "text": " ".join(str(token.get("text", "")) for token in tokens),
                }
            )
        x_lines = table.get("x_lines", [])
        tables.append(
            {
                "table_id": str(table.get("table_id", "")),
                "physical_columns": max(len(x_lines) - 1, 0),
                "cells": cells,
                "structure_issues": table.get("structure_issues", []),
            }
        )
    return tables


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip()[1:-1].split("|")]


def _align_table_columns(
    markdown: str, table_geometry: list[dict[str, object]]
) -> tuple[str, list[str]]:
    """Qwen의 행은 유지하고 colspan으로 접힌 물리 열만 빈칸으로 복원한다."""

    lines = markdown.splitlines()
    output: list[str] = []
    issues: list[str] = []
    table_index = 0
    index = 0
    while index < len(lines):
        if not (
            lines[index].strip().startswith("|") and lines[index].strip().endswith("|")
        ):
            output.append(lines[index])
            index += 1
            continue
        table_lines = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                break
            table_lines.append(lines[index])
            index += 1
        if table_index >= len(table_geometry):
            output.extend(table_lines)
            table_index += 1
            continue

        geometry = table_geometry[table_index]
        table_index += 1
        physical_columns = int(geometry.get("physical_columns", 0))
        markdown_columns = [len(_table_cells(line)) for line in table_lines]
        target_columns = max([physical_columns, *markdown_columns], default=0)
        if target_columns <= 0:
            output.extend(table_lines)
            continue
        # 이 문서의 복합 표는 첫 `구분` 셀이 두 개 이상의 좌측 열을 덮는 형태다.
        # 검출 셀에서 colspan으로 생기는 삽입 위치를 수집해 접힌 Markdown 행에 적용한다.
        insertion_positions = sorted(
            {
                int(cell.get("column", 0)) + 1
                for cell in geometry.get("cells", [])
                if isinstance(cell, dict) and int(cell.get("colspan", 1)) > 1
            }
        )
        if not insertion_positions:
            insertion_positions = [1]
        aligned_lines = []
        for row_index, line in enumerate(table_lines):
            values = _table_cells(line)
            if len(values) > target_columns:
                issues.append(
                    f"table {table_index} row {row_index}: markdown cells={len(values)}, "
                    f"target columns={target_columns}"
                )
                aligned_lines.append(line)
                continue
            for position in insertion_positions:
                if len(values) >= target_columns:
                    break
                values.insert(min(position, len(values)), "")
            while len(values) < target_columns:
                values.insert(min(insertion_positions[-1], len(values)), "")
            if len(values) != target_columns:
                issues.append(
                    f"table {table_index} row {row_index}: cannot align "
                    f"{len(values)} cells to {target_columns} columns"
                )
                aligned_lines.append(line)
                continue
            if row_index == 1:
                values = [":---"] * target_columns
            aligned_lines.append("| " + " | ".join(values) + " |")
        output.extend(aligned_lines)
    return "\n".join(output).strip() + "\n", issues


def _generate_json(
    base_url: str,
    model: str,
    image_paths: list[Path],
    prompt: str,
    schema: dict[str, object],
    num_predict: int,
) -> tuple[dict[str, object], dict[str, object]]:
    images_b64 = [
        base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths
    ]
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "images": images_b64,
                "stream": False,
                "think": False,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 32768,
                    "num_predict": num_predict,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Ollama restoration request failed: {error}") from error
    if payload.get("done_reason") == "length":
        raise RuntimeError("Qwen restoration output exceeded the token limit")
    raw = payload.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Qwen returned an empty restoration result")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Qwen returned an invalid structured object")
    metadata = {
        "total_duration": payload.get("total_duration"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
    }
    return value, metadata


def _generate_markdown(
    base_url: str,
    model: str,
    image_paths: list[Path],
    prompt: str,
) -> tuple[dict[str, object], dict[str, object]]:
    value, metadata = _generate_json(
        base_url,
        model,
        image_paths,
        prompt,
        _output_schema(),
        num_predict=12288,
    )
    if not isinstance(value.get("markdown"), str):
        raise TypeError("Qwen returned an invalid restoration object")
    return value, metadata


def _validate_markdown(
    markdown: str,
    viewer_text: str,
    ocr_record: dict[str, object],
    audit: dict[str, object] | None = None,
) -> dict[str, object]:
    ocr_text = " ".join(
        str(item.get("text", ""))
        for item in ocr_record.get("tokens", [])
        if isinstance(item, dict)
    )
    source_text = f"{viewer_text}\n{ocr_text}"
    source_numbers = {str(int(value)) for value in NUMBER_PATTERN.findall(source_text)}
    output_numbers = {str(int(value)) for value in NUMBER_PATTERN.findall(markdown)}
    unsupported_numbers = sorted(output_numbers - source_numbers, key=int)

    source_normalized = _normalize(source_text)
    # <br>은 표 셀 안의 줄바꿈 표현이지 문서에서 생성한 단어가 아니다.
    markdown_text = re.sub(r"<[^>]+>", " ", markdown)
    output_words = sorted(set(WORD_PATTERN.findall(markdown_text)))
    unsupported_words = [
        word
        for word in output_words
        if word not in MARKDOWN_STRUCTURE_WORDS
        and _normalize(word) not in source_normalized
    ]
    supported_count = len(output_words) - len(unsupported_words)
    coverage = supported_count / len(output_words) if output_words else 1.0
    output_normalized = _normalize(markdown_text)
    source_words = sorted(set(WORD_PATTERN.findall(viewer_text)))
    missing_source_words = [
        word for word in source_words if _normalize(word) not in output_normalized
    ]
    source_coverage = (
        (len(source_words) - len(missing_source_words)) / len(source_words)
        if source_words
        else 1.0
    )
    table_issues: list[str] = []
    expected_columns: int | None = None
    in_table = False
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            in_table = False
            expected_columns = None
            continue
        column_count = len(stripped.split("|")) - 2
        if not in_table:
            expected_columns = column_count
            in_table = True
        elif column_count != expected_columns:
            table_issues.append(
                f"line {line_number}: expected {expected_columns} columns, "
                f"found {column_count}"
            )

    audit = audit if isinstance(audit, dict) else {}
    audit_issues = [
        str(issue) for issue in audit.get("issues", []) if str(issue).strip()
    ]
    for key in (
        "all_visible_text_transcribed",
        "table_rows_aligned",
        "merged_cells_preserved",
        "no_added_content",
    ):
        if audit.get(key) is not True:
            audit_issues.append(f"Qwen audit failed: {key}")

    return {
        "numbers_grounded": not unsupported_numbers,
        "unsupported_numbers": unsupported_numbers,
        "word_coverage": round(coverage, 6),
        "unsupported_words": unsupported_words,
        "source_word_coverage": round(source_coverage, 6),
        "missing_source_words": missing_source_words,
        "table_issues": table_issues,
        "audit_issues": audit_issues,
        "status": (
            "generated"
            if not unsupported_numbers
            and not unsupported_words
            and source_coverage >= 0.98
            and not table_issues
            else "needs_review"
        ),
    }


def _correct_grounded_words(
    markdown: str, viewer_text: str, ocr_record: dict[str, object]
) -> tuple[str, list[dict[str, object]]]:
    """원문에 없는 단어가 유일한 근접 원문 단어를 가질 때만 오탈자를 교정한다."""

    source_text = (
        viewer_text
        + " "
        + " ".join(
            str(item.get("text", ""))
            for item in ocr_record.get("tokens", [])
            if isinstance(item, dict)
        )
    )
    source_normalized = _normalize(source_text)
    candidates = sorted(set(WORD_PATTERN.findall(source_text)))
    output_words = sorted(set(WORD_PATTERN.findall(re.sub(r"<[^>]+>", " ", markdown))))
    corrections = []
    corrected = markdown
    for word in output_words:
        if _normalize(word) in source_normalized:
            continue
        ranked = sorted(
            (
                (difflib.SequenceMatcher(None, word, candidate).ratio(), candidate)
                for candidate in candidates
            ),
            reverse=True,
        )
        if not ranked:
            continue
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < 0.75 or second_score > 0.67:
            continue
        pattern = re.compile(rf"(?<![A-Za-z가-힣]){re.escape(word)}(?![A-Za-z가-힣])")
        corrected, count = pattern.subn(best, corrected)
        if count:
            corrections.append(
                {
                    "from": word,
                    "to": best,
                    "similarity": round(best_score, 6),
                    "occurrences": count,
                }
            )
    return corrected, corrections


def _prepare_generated_markdown(
    markdown: str,
    table_geometry: list[dict[str, object]],
    viewer_text: str,
    ocr_record: dict[str, object],
    audit: dict[str, object] | None,
) -> tuple[str, dict[str, object]]:
    cleaned = _clean_markdown(markdown)
    aligned, alignment_issues = _align_table_columns(cleaned, table_geometry)
    corrected, word_corrections = _correct_grounded_words(
        aligned, viewer_text, ocr_record
    )
    validation = _validate_markdown(corrected, viewer_text, ocr_record, audit)
    validation["grounded_word_corrections"] = word_corrections
    validation["table_alignment_issues"] = alignment_issues
    if not alignment_issues and not validation.get("table_issues"):
        validation["audit_issues"] = [
            issue
            for issue in validation.get("audit_issues", [])
            if issue != "Qwen audit failed: merged_cells_preserved"
        ]
        if (
            not validation.get("unsupported_numbers")
            and not validation.get("unsupported_words")
            and validation.get("source_word_coverage", 0) >= 0.98
        ):
            validation["status"] = "generated"
    if alignment_issues:
        validation["status"] = "needs_review"
    return corrected, validation


def _validation_messages(validation: dict[str, object]) -> list[str]:
    """자동 재검증 프롬프트에 넣을 짧은 오류 목록을 만든다."""

    table_issues = [str(item) for item in validation.get("table_issues", [])]
    issues = list(table_issues)
    issues.extend(str(item) for item in validation.get("table_alignment_issues", []))
    column_counts: list[int] = []
    for issue in table_issues:
        column_counts.extend(
            int(value)
            for value in re.findall(r"(?:expected|found) (\d+) columns", issue)
        )
    if column_counts:
        issues.append(
            f"이 표의 논리 열 수는 {max(column_counts)}개다. header, separator, "
            f"첫 행부터 마지막 행까지 모두 정확히 {max(column_counts)}열로 다시 쓰고 "
            "4열 행을 남기지 않는다. 병합 셀이 덮는 위치는 빈칸으로 둔다."
        )
    issues.extend(str(item) for item in validation.get("audit_issues", []))
    unsupported_numbers = validation.get("unsupported_numbers", [])
    unsupported_words = validation.get("unsupported_words", [])
    missing_source_words = validation.get("missing_source_words", [])
    if unsupported_numbers:
        issues.append(f"원문에서 확인되지 않는 숫자: {unsupported_numbers}")
    if unsupported_words:
        issues.append(f"원문에서 확인되지 않는 단어: {unsupported_words}")
    if validation.get("source_word_coverage", 0) < 0.98:
        issues.append(f"Markdown에서 누락된 원문 단어: {missing_source_words}")
    return issues


def _has_repairable_issue(validation: dict[str, object]) -> bool:
    """재호출로 고칠 수 있고 코드로 재검증 가능한 오류만 판정한다."""

    return validation.get("source_word_coverage", 0) < 0.98 or any(
        validation.get(key)
        for key in (
            "unsupported_numbers",
            "unsupported_words",
            "table_issues",
            "table_alignment_issues",
        )
    )


def _uncertainties_for(markdown: str, generated: dict[str, object]) -> list[object]:
    """실제 `[불확실]` 표기가 없는 모델의 설명성 uncertainty는 버린다."""

    if "[불확실]" not in markdown:
        return []
    value = generated.get("uncertainties", [])
    return value if isinstance(value, list) else []


def _word_correction_hints(
    validation: dict[str, object], viewer_text: str, ocr_record: dict[str, object]
) -> list[str]:
    """Qwen이 오탈자를 고칠 때 참고할 원문 유사어만 제시한다."""

    source_text = (
        viewer_text
        + " "
        + " ".join(
            str(item.get("text", ""))
            for item in ocr_record.get("tokens", [])
            if isinstance(item, dict)
        )
    )
    candidates = sorted(set(WORD_PATTERN.findall(source_text)))
    hints = []
    for word in validation.get("unsupported_words", []):
        matches = difflib.get_close_matches(str(word), candidates, n=3, cutoff=0.65)
        if matches:
            hints.append(f"{word!r}의 원문 후보: {matches}")
    return hints


def _page_record(
    page: dict[str, object],
    pdf_path: Path,
    model: str,
    model_digest: str,
    base_url: str,
    force: bool,
) -> dict[str, object]:
    pdf_page = int(page["viewer_page"])
    printed_page = int(page["printed_page"])
    image_path = _render_pdf_page(pdf_path, pdf_page)
    image_paths = _page_images(image_path)
    ocr_path = CACHE_ROOT / "ocr" / f"page-{pdf_page:03d}.json"
    viewer_path = RAW_ROOT / str(page["text_path"])
    if not image_path.is_file() or not ocr_path.is_file() or not viewer_path.is_file():
        raise FileNotFoundError(f"Page {pdf_page} restoration sources are missing")

    viewer_text = viewer_path.read_text(encoding="utf-8")
    ocr_record = _read_json(ocr_path)
    table_geometry = _table_geometry(pdf_page, ocr_record)
    cache_path = CACHE_ROOT / "restoration" / f"page-{pdf_page:03d}.json"
    source_input_hash = _sha256_text(
        json.dumps(
            {
                "images": [_sha256_bytes(path.read_bytes()) for path in image_paths],
                "viewer_text": _sha256_text(viewer_text),
                "ocr": _sha256_bytes(ocr_path.read_bytes()),
                "table_geometry": table_geometry,
                "restoration_version": RESTORATION_VERSION,
                "model_digest": model_digest,
            },
            sort_keys=True,
        )
    )
    if cache_path.exists() and not force:
        cached = _read_json(cache_path)
        if cached.get("source_input_hash") == source_input_hash:
            return cached

    prompt = _prompt(printed_page, table_geometry)
    input_hash = _sha256_text(
        json.dumps(
            {
                "source_input_hash": source_input_hash,
                "table_geometry": table_geometry,
                "prompt": _sha256_text(prompt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    draft, draft_metadata = _generate_markdown(base_url, model, image_paths, prompt)
    draft_markdown, draft_validation = _prepare_generated_markdown(
        str(draft["markdown"]),
        table_geometry,
        viewer_text,
        ocr_record,
        draft.get("audit"),
    )
    if draft_validation["status"] == "generated" and not _uncertainties_for(
        draft_markdown, draft
    ):
        generated = draft
        markdown = draft_markdown
        validation = draft_validation
        verification_prompt = ""
        verification_metadata = None
    else:
        verification_prompt = _verification_prompt(
            printed_page, table_geometry, draft_markdown
        )
        generated, verification_metadata = _generate_markdown(
            base_url, model, image_paths, verification_prompt
        )
        markdown, validation = _prepare_generated_markdown(
            str(generated["markdown"]),
            table_geometry,
            viewer_text,
            ocr_record,
            generated.get("audit"),
        )
    repair_metadata: list[dict[str, object]] = []
    for _ in range(MAX_REPAIR_ATTEMPTS):
        if not _has_repairable_issue(validation):
            break
        repair_prompt = _verification_prompt(
            printed_page,
            table_geometry,
            markdown,
            _validation_messages(validation)
            + _word_correction_hints(validation, viewer_text, ocr_record),
        )
        generated, metadata = _generate_markdown(
            base_url, model, image_paths, repair_prompt
        )
        repair_metadata.append(metadata)
        markdown, validation = _prepare_generated_markdown(
            str(generated["markdown"]),
            table_geometry,
            viewer_text,
            ocr_record,
            generated.get("audit"),
        )
    uncertainties = _uncertainties_for(markdown, generated)
    if uncertainties:
        validation["status"] = "needs_review"
    record = {
        "schema_version": SCHEMA_VERSION,
        "restoration_version": RESTORATION_VERSION,
        "document_id": "2026-2-course-main-book",
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "table_geometry": table_geometry,
        "markdown": markdown,
        "uncertainties": uncertainties,
        "validation": validation,
        "review_status": "pending",
        "source": {
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "image_sha256": _sha256_bytes(image_path.read_bytes()),
            "detail_image_paths": [
                path.relative_to(ROOT).as_posix() for path in image_paths[1:]
            ],
            "viewer_text_path": viewer_path.relative_to(ROOT).as_posix(),
            "viewer_text_sha256": _sha256_text(viewer_text),
            "ocr_path": ocr_path.relative_to(ROOT).as_posix(),
            "ocr_sha256": _sha256_bytes(ocr_path.read_bytes()),
        },
        "generation": {
            "model": model,
            "model_digest": model_digest,
            "prompt_hash": _sha256_text(prompt),
            "verification_prompt_hash": (
                _sha256_text(verification_prompt) if verification_prompt else None
            ),
            "draft_markdown_sha256": _sha256_text(draft_markdown),
            "verification_changed": draft_markdown != markdown,
            "draft": draft_metadata,
            "verification": verification_metadata,
            "repairs": repair_metadata,
        },
        "source_input_hash": source_input_hash,
        "input_hash": input_hash,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(cache_path, record)
    return record


def _write_outputs(
    records: list[dict[str, object]],
    model: str,
    model_digest: str,
    failures: list[dict[str, object]] | None = None,
) -> None:
    pages_root = RESTORED_ROOT / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        page_path = pages_root / f"page-{int(record['pdf_page']):03d}.md"
        page_path.write_text(str(record["markdown"]), encoding="utf-8")

    all_cache_records = sorted(
        (
            record
            for path in (CACHE_ROOT / "restoration").glob("page-*.json")
            if (record := _read_json(path)).get("restoration_version")
            == RESTORATION_VERSION
            and record.get("generation", {}).get("model_digest") == model_digest
        ),
        key=lambda item: int(item["pdf_page"]),
    )
    _write_jsonl(RESTORED_ROOT / "document.jsonl", all_cache_records)
    combined = "\n\n".join(
        f"<!-- pdf_page={item['pdf_page']} printed_page={item['printed_page']} -->\n\n"
        f"{str(item['markdown']).strip()}"
        for item in all_cache_records
    )
    (RESTORED_ROOT / "document.md").write_text(combined + "\n", encoding="utf-8")
    _write_json(
        RESTORED_ROOT / "manifest.json",
        {
            "schema_version": "restored-document-manifest-v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "model": model,
            "model_digest": model_digest,
            "pages": [int(item["pdf_page"]) for item in all_cache_records],
            "failures": failures or [],
            "counts": {
                "pages": len(all_cache_records),
                "generated": sum(
                    item.get("validation", {}).get("status") == "generated"
                    for item in all_cache_records
                ),
                "needs_review": sum(
                    item.get("validation", {}).get("status") == "needs_review"
                    for item in all_cache_records
                ),
                "pending_human_review": sum(
                    item.get("review_status") == "pending" for item in all_cache_records
                ),
                "failed": len(failures or []),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore page-level Markdown from the HYU course-guide PDF"
    )
    parser.add_argument("--pages", default="1-75", help="Printed pages, e.g. 1-4,40")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _read_json(RAW_ROOT / "manifest.json")
    pdf_path = RAW_ROOT / str(manifest.get("pdf", {}).get("path", ""))
    if not pdf_path.is_file():
        raise SystemExit(f"Source PDF is missing: {pdf_path}")
    page_manifest = {
        int(page["printed_page"]): page
        for page in manifest.get("pages", [])
        if isinstance(page, dict) and page.get("printed_page") is not None
    }
    requested = _pages(args.pages)
    missing = [page for page in requested if page not in page_manifest]
    if missing:
        raise SystemExit(f"Printed pages are outside this book: {missing}")
    digest = _model_digest(args.ollama_url, args.model)
    records = []
    failures: list[dict[str, object]] = []
    for index, printed_page in enumerate(requested, start=1):
        page = page_manifest[printed_page]
        print(
            f"[{index}/{len(requested)}] restore printed page {printed_page} "
            f"(PDF {page['viewer_page']})",
            flush=True,
        )
        try:
            records.append(
                _page_record(
                    page, pdf_path, args.model, digest, args.ollama_url, args.force
                )
            )
        # 모델·이미지·검증 단계 중 어느 하나가 실패해도 다음 페이지는 계속 복원한다.
        except Exception as error:  # noqa: BLE001
            failure = {
                "printed_page": printed_page,
                "pdf_page": int(page["viewer_page"]),
                "error": f"{type(error).__name__}: {error}",
                "created_at": datetime.now(UTC).isoformat(),
            }
            failures.append(failure)
            _write_json(
                CACHE_ROOT
                / "restoration-errors"
                / f"page-{int(page['viewer_page']):03d}.json",
                failure,
            )
            print(f"  failed: {failure['error']}", flush=True)
    _write_outputs(records, args.model, digest, failures)
    print(
        f"Restored {len(records)} page(s) into {RESTORED_ROOT.relative_to(ROOT)}",
        flush=True,
    )
    if failures:
        print(
            f"Failed pages: {len(failures)} (rerun the same command to retry)",
            flush=True,
        )


if __name__ == "__main__":
    main()
