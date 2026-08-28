"""초안 생성과 통합 GUI, 두 명령만 제공하는 CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import CANONICAL_DOCUMENT, OLLAMA_URL, TOOL_ROOT
from pipeline.service import generate_run, run_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ontology population for the 2026-2 HYU course-registration guide"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate",
        help="Download upstream ontologies, segment the document and create draft TTL",
    )
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--input", type=Path, default=CANONICAL_DOCUMENT)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--ollama-url", default=OLLAMA_URL)

    review = commands.add_parser(
        "review",
        help="Open the integrated reconciliation, validation and publication GUI",
    )
    review.add_argument("--run-id", required=True)
    review.add_argument("--ollama-url", default=OLLAMA_URL)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _launch_review(run_id: str, ollama_url: str) -> int:
    # GUI도 CLI와 같은 run ID 검증을 사용해 경로 이탈을 막는다.
    run_root(run_id)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(TOOL_ROOT / "review" / "app.py"),
        "--",
        "--run-id",
        run_id,
        "--ollama-url",
        ollama_url,
    ]
    return subprocess.run(command, check=False, cwd=TOOL_ROOT).returncode


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "generate":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            result = generate_run(
                args.run_id,
                resume=args.resume,
                canonical_path=args.input,
                limit=args.limit,
                ollama_url=args.ollama_url,
            )
            _print(result)
            # partial/failed 페이지는 GUI에서 후검수할 작업 목록이다. 생성 명령은
            # 가능한 페이지를 끝까지 보존했으면 비정상 종료로 취급하지 않는다.
        elif args.command == "review":
            raise SystemExit(_launch_review(args.run_id, args.ollama_url))
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
