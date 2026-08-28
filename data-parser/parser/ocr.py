from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def extract_page_image(pdf_path: Path, pdf_page: int, output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp_dir:
        prefix = Path(temp_dir) / "page"
        subprocess.run(
            [
                "pdfimages",
                "-f",
                str(pdf_page),
                "-l",
                str(pdf_page),
                "-j",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        images = list(Path(temp_dir).glob("page-*"))
        if len(images) != 1:
            raise RuntimeError(
                f"Expected one embedded image on PDF page {pdf_page}, found {len(images)}"
            )
        shutil.move(images[0], output_path)
    return output_path


def _bbox(box: object) -> list[int]:
    values = box.tolist() if hasattr(box, "tolist") else list(box)
    if len(values) == 4 and not isinstance(values[0], list):
        return [int(value) for value in values]
    xs = [int(point[0]) for point in values]
    ys = [int(point[1]) for point in values]
    return [min(xs), min(ys), max(xs), max(ys)]


def _viewer_match(text: str, viewer_text: str) -> bool:
    normalized = normalize_text(text)
    return bool(normalized) and normalized in normalize_text(viewer_text)


def run_ocr(
    image_path: Path,
    pdf_page: int,
    printed_page: int,
    viewer_text: str,
    cache_path: Path,
    model_root: Path,
    force: bool = False,
) -> dict[str, object]:
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("source", {}).get("image_sha256") == image_sha256:
            return cached

    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(model_root))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        lang="korean",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device="cpu",
    )
    results = list(engine.predict(str(image_path)))
    if len(results) != 1:
        raise RuntimeError(f"Expected one OCR page result, found {len(results)}")
    result = results[0]

    tokens: list[dict[str, object]] = []
    token_number = 0
    for line_index in range(len(result["rec_texts"])):
        line_id = f"p{pdf_page:03d}-l{line_index + 1:04d}"
        words = result["text_word"][line_index] or [result["rec_texts"][line_index]]
        boxes = result["text_word_boxes"][line_index]
        if len(words) != len(boxes):
            words = [result["rec_texts"][line_index]]
            boxes = [result["rec_boxes"][line_index]]
        for word_index, (word, box) in enumerate(zip(words, boxes, strict=True), start=1):
            text = str(word).strip()
            if not text:
                continue
            token_number += 1
            tokens.append(
                {
                    "token_id": f"p{pdf_page:03d}-t{token_number:04d}",
                    "line_id": line_id,
                    "line_word_index": word_index,
                    "text": text,
                    "bbox": _bbox(box),
                    "confidence": round(float(result["rec_scores"][line_index]), 6),
                    "viewer_text_match": _viewer_match(text, viewer_text),
                }
            )

    record = {
        "document_id": "2026-2-course-main-book",
        "pdf_page": pdf_page,
        "printed_page": printed_page,
        "image_size": {"width": 1018, "height": 1440},
        "tokens": tokens,
        "source": {
            "image_path": image_path.as_posix(),
            "image_sha256": image_sha256,
            "viewer_text_sha256": hashlib.sha256(viewer_text.encode("utf-8")).hexdigest(),
        },
        "ocr": {
            "engine": "PaddleOCR",
            "version": "3.7.0",
            "detection_model": "PP-OCRv5_server_det",
            "recognition_model": "korean_PP-OCRv5_mobile_rec",
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
