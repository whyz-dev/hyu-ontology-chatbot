"""Append-only 검수 이력에서 현재 후보와 정합 결정 상태를 읽는다."""

from __future__ import annotations

from pathlib import Path

from adapters.storage import latest_by, read_jsonl, sha256_text, stable_json


def population_decision_path(run_root: Path) -> Path:
    return run_root / "population-decisions.jsonl"


def candidate_hash(candidate: dict[str, object]) -> str:
    """검수한 후보가 이후 바뀌었는지 판별하는 안정적인 snapshot hash다."""
    return sha256_text(stable_json(candidate))


def population_decisions(run_root: Path) -> dict[str, dict[str, object]]:
    path = population_decision_path(run_root)
    return latest_by(read_jsonl(path), "unit_id") if path.exists() else {}


def current_population_decisions(
    run_root: Path,
    candidates: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """현재 candidate hash와 일치하는 페이지 결정만 돌려준다."""
    decisions = population_decisions(run_root)
    return {
        unit_id: decision
        for unit_id, decision in decisions.items()
        if unit_id in candidates
        and decision.get("candidate_hash") == candidate_hash(candidates[unit_id])
    }


def reviewed_population_candidates(
    run_root: Path,
    units: dict[str, dict[str, object]],
    candidates: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """현재 생성물 중 사람이 승인한 완전한 페이지 후보만 투영한다."""
    decisions = current_population_decisions(run_root, candidates)
    accepted: dict[str, dict[str, object]] = {}
    issues: list[str] = []
    for unit_id in units:
        candidate = candidates.get(unit_id)
        if candidate is None:
            issues.append(f"{unit_id}: candidate is missing")
            continue
        if (
            candidate.get("status") != "success"
            or not isinstance(candidate.get("coverage"), dict)
            or candidate["coverage"].get("complete") is not True
        ):
            issues.append(f"{unit_id}: latest candidate is not complete")
            continue
        decision = decisions.get(unit_id)
        if decision is None:
            issues.append(f"{unit_id}: current candidate is not reviewed")
            continue
        if decision.get("decision") == "reject":
            issues.append(f"{unit_id}: current candidate was rejected")
            continue
        if decision.get("decision") not in {"accept", "amend"}:
            issues.append(
                f"{unit_id}: current candidate has an invalid review decision"
            )
            continue
        accepted[unit_id] = candidate
    return accepted, issues


def alignment_decision_path(run_root: Path) -> Path:
    return run_root / "alignment-decisions.jsonl"


def alignment_decisions(run_root: Path) -> dict[str, dict[str, object]]:
    path = alignment_decision_path(run_root)
    return latest_by(read_jsonl(path), "alignment_id") if path.exists() else {}
