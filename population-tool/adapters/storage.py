"""재현 가능한 hash 계산과 JSON/JSONL 파일 입출력을 제공한다."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    # ID와 hash가 키 순서나 불필요한 공백에 따라 달라지지 않게 한다.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def write_json(path: Path, value: object) -> None:
    # 같은 디렉터리의 임시 파일을 원자적으로 교체해 반쯤 써진 산출물을 막는다.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    # 검수 이력과 checkpoint는 기존 행을 고치지 않는 append-only 계약이다.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def latest_by(
    records: Iterable[dict[str, object]], key: str
) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        identifier = record.get(key)
        if isinstance(identifier, str):
            latest[identifier] = record
    return latest
