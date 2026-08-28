from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from generation import (
    QUESTION_TYPES,
    SINGLE_EVIDENCE_QUESTION_TYPES,
    build_prompt,
    flatten_bundles,
    generate_round,
)
from sources import (
    extract_evidence_units,
    load_hyu_context,
    load_uvabot_questions,
    parse_pages,
    select_diverse_questions,
    select_evidence_units,
)
from validation import (
    finalize_records,
    select_distinct_records,
    write_jsonl,
    write_preview,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLES_PER_ROUND = 10
QUESTIONS_PER_ROUND = BUNDLES_PER_ROUND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse HYU course-registration conversations"
    )
    parser.add_argument(
        "--uvabot",
        type=Path,
        default=(
            ROOT
            / "data"
            / "external"
            / "qa"
            / "uvabot"
            / "UVaBotMistakes.csv"
        ),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=ROOT / "data" / "processed" / "preview.md",
    )
    parser.add_argument(
        "--notice",
        type=Path,
        default=ROOT / "data" / "cache" / "notice.json",
    )
    parser.add_argument(
        "--pages",
        default="2-16",
        help="Printed pages used as context; use 'all' for the whole guide",
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument(
        "--extra-rounds",
        type=int,
        default=4,
        help="Over-generate rounds before strict semantic/evidence deduplication",
    )
    parser.add_argument(
        "--round-offset",
        type=int,
        default=0,
        help="Start diversity rounds after this many completed rounds",
    )
    parser.add_argument("--style-examples", type=int, default=30)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=131072)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "qa",
    )
    return parser.parse_args()


def _rotated_styles(
    pool: list[dict[str, str]],
    round_index: int,
    count: int,
) -> list[dict[str, str]]:
    start = (round_index * count) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(count)]


def _preferred_types(round_index: int) -> list[str]:
    shift = (round_index * 7) % len(SINGLE_EVIDENCE_QUESTION_TYPES)
    return [
        SINGLE_EVIDENCE_QUESTION_TYPES[
            (shift + offset) % len(SINGLE_EVIDENCE_QUESTION_TYPES)
        ]
        for offset in range(BUNDLES_PER_ROUND)
    ]


def _dialogue_modes(round_index: int) -> list[str]:
    modes = [
        "standalone",
        "standalone",
        "standalone",
        "standalone",
        "standalone",
        "standalone",
        "clarification",
        "contextual_followup",
        "user_correction",
        "confirmation",
    ]
    shift = round_index % len(modes)
    return modes[shift:] + modes[:shift]


def main() -> None:
    args = parse_args()
    if args.target <= 0:
        raise SystemExit("--target must be positive")

    questions = load_uvabot_questions(args.uvabot)
    style_pool_size = min(len(questions), max(args.style_examples * 4, 120))
    style_pool = select_diverse_questions(questions, style_pool_size)
    context = load_hyu_context(args.preview, args.notice, parse_pages(args.pages))
    target_rounds = math.ceil(args.target / QUESTIONS_PER_ROUND)
    rounds = target_rounds + args.extra_rounds
    evidence_units = select_evidence_units(
        extract_evidence_units(context), rounds * QUESTIONS_PER_ROUND
    )
    print(
        f"Generating {rounds * QUESTIONS_PER_ROUND} candidates in {rounds} "
        f"round(s), then selecting {args.target} fact-distinct conversations with "
        f"pre-assigned unique evidence units",
        flush=True,
    )

    candidates: list[dict[str, object]] = []
    valid_style_ids: set[str] = set()
    for local_round_index in range(rounds):
        round_index = args.round_offset + local_round_index
        round_number = round_index + 1
        styles = _rotated_styles(style_pool, round_index, args.style_examples)
        valid_style_ids.update(item["source_id"] for item in styles)
        preferred_types = _preferred_types(round_index)
        dialogue_modes = _dialogue_modes(round_index)
        evidence_start = local_round_index * BUNDLES_PER_ROUND
        evidence_assignments = evidence_units[
            evidence_start : evidence_start + BUNDLES_PER_ROUND
        ]
        page_summary = ", ".join(
            sorted({item["locator"] for item in evidence_assignments})
        )
        print(
            f"- round {local_round_index + 1}/{rounds} "
            f"(diversity round {round_number}): {page_summary}",
            flush=True,
        )
        prompt = build_prompt(
            styles,
            round_number,
            evidence_assignments,
            dialogue_modes,
            preferred_types,
        )
        bundles, metadata = generate_round(
            args.ollama_url.rstrip("/"),
            args.model,
            prompt,
            args.num_ctx,
            round_number,
            dialogue_modes,
        )
        round_candidates = flatten_bundles(
            bundles,
            metadata,
            evidence_assignments,
            dialogue_modes,
        )
        candidates.extend(round_candidates)

    candidate_records, _ = finalize_records(
        candidates,
        context,
        valid_style_ids,
        {"model": args.model},
    )
    candidate_issue_counts = Counter(
        issue
        for record in candidate_records
        for issue in record["automatic_validation"]["issues"]
    )
    candidate_passed = sum(
        record["automatic_validation"]["status"] == "passed"
        for record in candidate_records
    )
    print(
        f"Candidate validation: {candidate_passed}/{len(candidate_records)} passed",
        flush=True,
    )
    for issue, count in candidate_issue_counts.most_common(8):
        print(f"  - {count}x {issue}", flush=True)
    records = select_distinct_records(candidate_records, args.target)
    flagged = [
        record
        for record in records
        if record["automatic_validation"]["status"] == "failed"
    ]
    dataset_path = args.output_dir / "dataset.jsonl"
    preview_path = args.output_dir / "preview.md"
    write_jsonl(dataset_path, records)
    write_preview(preview_path, records)

    type_counts = Counter(record["question_type"] for record in records)
    missing_types = sorted(set(QUESTION_TYPES) - set(type_counts))
    print(
        f"Wrote {len(records)} conversations with {len(type_counts)} question types; "
        f"flagged {len(flagged)} for review",
        flush=True,
    )
    print(
        "Type range: "
        f"{min(type_counts.values(), default=0)}-{max(type_counts.values(), default=0)}",
        flush=True,
    )
    if missing_types:
        print(f"Missing types: {', '.join(missing_types)}", flush=True)


if __name__ == "__main__":
    main()
