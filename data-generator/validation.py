from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from generation import QUESTION_TYPES


FOREIGN_TERMS = re.compile(
    r"universidad|valladolid|\buva\b|castilla|segovia|palencia|oviedo|cantabria",
    re.IGNORECASE,
)


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", value)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _last_user_text(record: dict[str, Any]) -> str:
    user_messages = [
        str(message.get("content", ""))
        for message in record.get("messages", [])
        if message.get("role") == "user"
    ]
    return user_messages[-1] if user_messages else ""


def _question_fingerprint(record: dict[str, Any]) -> str:
    return _hash(re.sub(r"[^0-9a-z가-힣]+", "", _last_user_text(record).casefold()))


def _query_ir_fingerprint(record: dict[str, Any]) -> str:
    query_ir = record.get("query_ir", {})
    canonical = json.dumps(query_ir, ensure_ascii=False, sort_keys=True)
    return _hash(_normalized(canonical).casefold())


def _evidence_signatures(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (
            str(item.get("source_unit_id", "")).strip(),
            _normalized(str(item.get("locator", ""))).casefold(),
            _normalized(str(item.get("quote", ""))).casefold(),
        )
        for item in record.get("evidence", [])
    ]


def _evidence_ids(record: dict[str, Any]) -> list[str]:
    return [
        source_unit_id or _hash(f"{locator}\n{quote}")
        for source_unit_id, locator, quote in _evidence_signatures(record)
    ]


def _evidence_bundle_fingerprint(record: dict[str, Any]) -> str:
    return _hash("\n".join(sorted(_evidence_ids(record))))


def _character_ngrams(value: str, size: int = 5) -> set[str]:
    if len(value) <= size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _same_evidence_fact(
    left: tuple[str, str, str],
    right: tuple[str, str, str],
) -> bool:
    left_id, left_locator, left_quote = left
    right_id, right_locator, right_quote = right
    if left_id and right_id:
        return left_id == right_id
    if left_locator != right_locator:
        return False
    if left_quote == right_quote or left_quote in right_quote or right_quote in left_quote:
        return True
    left_ngrams = _character_ngrams(left_quote)
    right_ngrams = _character_ngrams(right_quote)
    union = left_ngrams | right_ngrams
    return bool(union) and len(left_ngrams & right_ngrams) / len(union) >= 0.72


def validate_conversation(
    record: dict[str, Any],
    context: str,
    valid_style_ids: set[str],
) -> list[str]:
    issues: list[str] = []
    question_type = record.get("question_type")
    if question_type not in QUESTION_TYPES:
        issues.append("unknown question_type")

    style_ids = record.get("style_source_ids", [])
    if not style_ids or any(item not in valid_style_ids for item in style_ids):
        issues.append("invalid UVaBot style source")

    messages = record.get("messages", [])
    if len(messages) < 2:
        issues.append("conversation has fewer than two messages")
    elif messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        issues.append("conversation must start with user and end with assistant")
    for before, after in zip(messages, messages[1:]):
        if before.get("role") == after.get("role"):
            issues.append("message roles do not alternate")
            break

    if messages:
        final_answer = _normalized(messages[-1].get("content", ""))
        reference_answer = _normalized(str(record.get("reference_answer", "")))
        if not final_answer or not reference_answer or (
            final_answer not in reference_answer and reference_answer not in final_answer
        ):
            issues.append("final assistant answer differs from reference_answer")

    combined_text = " ".join(str(item.get("content", "")) for item in messages)
    combined_text += " " + str(record.get("reference_answer", ""))
    if FOREIGN_TERMS.search(combined_text):
        issues.append("foreign university content leaked into HYU conversation")
    if re.search(r"[¿¡]", combined_text):
        issues.append("Spanish punctuation remains")
    if re.search(r"검수\s*필요|자동\s*승인|OCR|마크다운", combined_text, re.IGNORECASE):
        issues.append("parser review metadata leaked into the conversation")

    dialogue_mode = str(record.get("dialogue_mode", "standalone"))
    contextual_modes = {
        "clarification",
        "contextual_followup",
        "user_correction",
        "confirmation",
    }
    if dialogue_mode == "standalone":
        if len(messages) != 2:
            issues.append("standalone conversation must contain one user-answer pair")
        if question_type in contextual_modes:
            issues.append("standalone conversation uses a contextual question type")
    elif dialogue_mode in contextual_modes:
        if len(messages) < 4:
            issues.append("context-dependent mode requires at least four messages")
        if question_type != dialogue_mode:
            issues.append("context-dependent mode and question_type differ")
    else:
        issues.append("unknown dialogue_mode")

    if dialogue_mode == "clarification" and len(messages) >= 2:
        if "?" not in messages[1].get("content", ""):
            issues.append("clarification must ask for a missing condition")
        if messages[1].get("content") != "어떤 조건을 기준으로 확인할까요?":
            issues.append("clarification assistant message is not normalized")
    elif dialogue_mode in contextual_modes and len(messages) >= 2:
        reference = str(record.get("reference_answer", ""))
        if _normalized(messages[1].get("content", "")) != _normalized(reference):
            issues.append("intermediate assistant message is not evidence-grounded")
    if dialogue_mode == "user_correction":
        user_text = " ".join(
            item.get("content", "")
            for item in messages[2:]
            if item.get("role") == "user"
        )
        if not re.search(r"아니|정정|잘못|다시|제가 말한", user_text):
            issues.append("user_correction has no explicit correction")

    query_ir = record.get("query_ir", {})
    required_ir = {"intent", "subject", "constraints", "requested_fields"}
    if not required_ir.issubset(query_ir):
        issues.append("query_ir is incomplete")
    if not isinstance(query_ir.get("constraints", {}), dict):
        issues.append("query_ir.constraints must be an object")
    if not isinstance(query_ir.get("requested_fields", []), list):
        issues.append("query_ir.requested_fields must be an array")

    evidence = record.get("evidence", [])
    if not evidence:
        issues.append("evidence is empty")
    normalized_context = _normalized(context)
    for item in evidence:
        quote = str(item.get("quote", "")).strip()
        locator = str(item.get("locator", "")).strip()
        if not quote or _normalized(quote) not in normalized_context:
            issues.append(f"evidence quote is not in source context: {locator}")
        if not re.search(r"인쇄 페이지 \d+|policy-notice-2341", locator):
            issues.append(f"invalid evidence locator: {locator}")

    reference = str(record.get("reference_answer", ""))
    evidence_text = " ".join(str(item.get("quote", "")) for item in evidence)
    if reference.strip() and _normalized(reference) not in _normalized(evidence_text):
        issues.append("reference_answer is not an exact evidence substring")
    answer_numbers = set(re.findall(r"\d+", reference))
    evidence_numbers = set(re.findall(r"\d+", evidence_text))
    unsupported_numbers = answer_numbers - evidence_numbers - {"2", "2026"}
    if unsupported_numbers:
        issues.append(
            "answer numbers are absent from evidence: "
            + ", ".join(sorted(unsupported_numbers))
        )
    if reference.strip().startswith("아니요") and re.search(r"(?<!불)가능", reference):
        issues.append("negative answer also states that the action is possible")

    user_text = " ".join(
        item.get("content", "") for item in messages if item.get("role") == "user"
    )
    stated_years = re.findall(r"저는\s*(\d)\s*학년", user_text)
    if stated_years and not re.search(rf"{stated_years[-1]}\s*학년", reference):
        issues.append("answer does not match the user's last stated year")
    return issues


def finalize_records(
    candidates: list[dict[str, Any]],
    context: str,
    valid_style_ids: set[str],
    generation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        candidate_generation = candidate.pop("_generation", generation)
        messages = candidate.get("messages", [])
        reference_answer = str(candidate.get("reference_answer", "")).strip()
        if messages and messages[-1].get("role") == "assistant" and reference_answer:
            messages[-1]["content"] = reference_answer
        issues = validate_conversation(candidate, context, valid_style_ids)
        question_type = str(candidate.get("question_type", ""))
        message_fingerprint = _normalized(
            " ".join(item.get("content", "") for item in candidate.get("messages", []))
        )
        if message_fingerprint in seen_texts:
            issues.append("duplicate conversation")
        serialized = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        identifier = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        record = {
            "schema_version": "hyu-conversation-qa-v2",
            "conversation_id": f"hyu-qa-{identifier}",
            **candidate,
            "distinctness": {
                "question_fingerprint": _question_fingerprint(candidate),
                "query_ir_fingerprint": _query_ir_fingerprint(candidate),
                "evidence_bundle_fingerprint": _evidence_bundle_fingerprint(candidate),
                "evidence_ids": _evidence_ids(candidate),
            },
            "scope": {
                "academic_year": 2026,
                "semester": 2,
                "campus": "서울",
                "domain": "course_registration",
            },
            "automatic_validation": {
                "status": "failed" if issues else "passed",
                "issues": issues,
            },
            "generation": candidate_generation,
            "review_status": "needs_human_review",
        }
        records.append(record)
        if issues:
            flagged.append(record)
        seen_texts.add(message_fingerprint)
    return records, flagged


def select_distinct_records(
    records: list[dict[str, Any]],
    target: int,
) -> list[dict[str, Any]]:
    def quality(record: dict[str, Any]) -> tuple[bool, int, int]:
        issues = [
            issue
            for issue in record["automatic_validation"]["issues"]
            if issue != "duplicate conversation"
        ]
        return (
            bool(issues),
            len(issues),
            int(record.get("generation", {}).get("round", 999)),
        )

    if target <= 0:
        raise ValueError("target must be positive")

    remaining = [
        record
        for record in records
        if record.get("automatic_validation", {}).get("status") == "passed"
    ]
    selected: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    used_query_irs: set[str] = set()
    used_evidence: list[tuple[str, str, str]] = []
    type_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    mode_weights = {
        "standalone": 6,
        "clarification": 1,
        "contextual_followup": 1,
        "user_correction": 1,
        "confirmation": 1,
    }

    while remaining and len(selected) < target:
        compatible: list[dict[str, Any]] = []
        for record in remaining:
            distinctness = record["distinctness"]
            if distinctness["question_fingerprint"] in used_questions:
                continue
            if distinctness["query_ir_fingerprint"] in used_query_irs:
                continue
            signatures = _evidence_signatures(record)
            if any(
                _same_evidence_fact(candidate, used)
                for candidate in signatures
                for used in used_evidence
            ):
                continue
            compatible.append(record)

        if not compatible:
            break
        def selection_rank(record: dict[str, Any]) -> tuple[Any, ...]:
            record_quality = quality(record)
            mode = str(record.get("dialogue_mode", "standalone"))
            return (
                record_quality[0],
                record_quality[1],
                mode_counts.get(mode, 0) / mode_weights.get(mode, 1),
                type_counts.get(str(record.get("question_type", "")), 0),
                record_quality[2],
                record["conversation_id"],
            )

        chosen = min(compatible, key=selection_rank)
        selected.append(chosen)
        remaining.remove(chosen)
        distinctness = chosen["distinctness"]
        used_questions.add(distinctness["question_fingerprint"])
        used_query_irs.add(distinctness["query_ir_fingerprint"])
        used_evidence.extend(_evidence_signatures(chosen))
        question_type = str(chosen.get("question_type", ""))
        type_counts[question_type] = type_counts.get(question_type, 0) + 1
        dialogue_mode = str(chosen.get("dialogue_mode", "standalone"))
        mode_counts[dialogue_mode] = mode_counts.get(dialogue_mode, 0) + 1

    if len(selected) < target:
        raise ValueError(
            f"Only {len(selected)} records have distinct questions, Query IRs, and "
            f"non-overlapping evidence while passing automatic validation; requested "
            f"{target}. Generate more rounds or add source facts."
        )

    for record in selected:
        issues = [
            issue
            for issue in record["automatic_validation"]["issues"]
            if issue != "duplicate conversation"
        ]
        record["automatic_validation"] = {
            "status": "failed" if issues else "passed",
            "issues": issues,
        }
    return sorted(selected, key=lambda record: record["conversation_id"])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_preview(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    passed_count = sum(
        record["automatic_validation"]["status"] == "passed" for record in records
    )
    flagged_count = len(records) - passed_count
    question_type_count = len({record["question_type"] for record in records})
    evidence_bundle_count = len(
        {
            record["distinctness"]["evidence_bundle_fingerprint"]
            for record in records
        }
    )
    query_ir_count = len(
        {record["distinctness"]["query_ir_fingerprint"] for record in records}
    )
    dialogue_mode_counts: dict[str, int] = {}
    for record in records:
        mode = str(record.get("dialogue_mode", "standalone"))
        dialogue_mode_counts[mode] = dialogue_mode_counts.get(mode, 0) + 1
    lines = [
        "# 2026-2 수강신청 대화형 질문 후보",
        "",
        f"- 전체 대화 후보: {len(records)}개",
        f"- 질문 유형: {question_type_count}개",
        f"- 고유 근거 묶음: {evidence_bundle_count}개",
        f"- 고유 Query IR: {query_ir_count}개",
        "- 대화 형식: "
        + ", ".join(
            f"{mode} {count}개"
            for mode, count in sorted(dialogue_mode_counts.items())
        ),
        f"- 자동 검증 통과: {passed_count}개",
        f"- 수정 필요: {flagged_count}개",
        "- 현재 상태: 사람 검수 필요",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record['question_type']}",
                "",
                f"- ID: `{record['conversation_id']}`",
                f"- UVaBot 표현 참고: {', '.join(record['style_source_ids'])}",
                f"- 자동 검증: {record['automatic_validation']['status']}",
                "",
            ]
        )
        for issue in record["automatic_validation"]["issues"]:
            lines.append(f"- ⚠️ {issue}")
        if record["automatic_validation"]["issues"]:
            lines.append("")
        for message in record["messages"]:
            speaker = "사용자" if message["role"] == "user" else "챗봇"
            lines.append(f"**{speaker}:** {message['content']}")
            lines.append("")
        lines.append(f"**근거 답:** {record['reference_answer']}")
        lines.append("")
        lines.append("**근거:**")
        lines.append("")
        for evidence in record["evidence"]:
            lines.append(f"- {evidence['locator']}: {evidence['quote']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
