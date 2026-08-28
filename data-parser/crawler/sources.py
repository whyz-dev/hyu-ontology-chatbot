from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


BOOK_ID = "KYWV5GB6H6ZE"
BOOK_URL = "https://book.hanyang.ac.kr/Viewer/2026_2_course_main_book"
BOOK_API_URL = f"https://book.hanyang.ac.kr/Viewer/getBookXML/{BOOK_ID}"
PAGE_TEXT_URL = (
    f"https://book.hanyang.ac.kr/Viewer/getPageWords/{BOOK_ID}/{{page_code}}"
)
NOTICE_URL = (
    "https://policy.hanyang.ac.kr/front/communication/notice/notice-view?id=2341"
)
RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
WEB_ROOT = RAW_ROOT / "web"
USER_AGENT = "hyu-chatbot-research/0.1"


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.relative_to(RAW_ROOT).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _page_code(page: dict[str, object]) -> str:
    code = page["code"]
    if not isinstance(code, dict) or not isinstance(code.get("0"), str):
        raise ValueError("Unexpected book page code schema")
    return code["0"]


def crawl() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    files: list[dict[str, object]] = []

    viewer_html = _download(BOOK_URL)
    files.append(_write(WEB_ROOT / "book" / "viewer.html", viewer_html))

    book_json = _download(BOOK_API_URL)
    pages = json.loads(book_json)
    if not isinstance(pages, list) or len(pages) != 79:
        raise ValueError(f"Expected 79 book pages, received {len(pages)}")
    files.append(_write(WEB_ROOT / "book" / "pages.json", book_json))

    page_manifest: list[dict[str, object]] = []
    for viewer_page, page in enumerate(pages, start=1):
        code = _page_code(page)
        image_url = "https:" + str(page["src"])
        text_url = PAGE_TEXT_URL.format(page_code=code)
        text_file = WEB_ROOT / "book" / "text" / f"{viewer_page:03d}-{code}.txt"
        text_entry = _write(text_file, _download(text_url))
        files.append(text_entry)
        page_manifest.append(
            {
                "viewer_page": viewer_page,
                "printed_page": viewer_page - 4 if viewer_page >= 5 else None,
                "page_code": code,
                "text_url": text_url,
                "image_url": image_url,
                "text_path": text_entry["path"],
            }
        )

    notice_html = _download(NOTICE_URL)
    files.append(_write(WEB_ROOT / "notice" / "notice-2341.html", notice_html))

    pdf_files = sorted((RAW_ROOT / "pdf").glob("*.pdf"))
    if len(pdf_files) != 1:
        raise ValueError(f"Expected one source PDF in {RAW_ROOT / 'pdf'}, found {len(pdf_files)}")
    pdf_file = pdf_files[0]

    manifest = {
        "fetched_at": fetched_at,
        "sources": {
            "book": BOOK_URL,
            "book_api": BOOK_API_URL,
            "notice": NOTICE_URL,
        },
        "book_id": BOOK_ID,
        "pdf": {
            "path": pdf_file.relative_to(RAW_ROOT).as_posix(),
            "size": pdf_file.stat().st_size,
            "sha256": hashlib.sha256(pdf_file.read_bytes()).hexdigest(),
        },
        "pages": page_manifest,
        "files": files,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    _write(RAW_ROOT / "manifest.json", manifest_bytes)
    print(f"Crawled text for {len(pages)} book pages and 1 notice into {WEB_ROOT}")
