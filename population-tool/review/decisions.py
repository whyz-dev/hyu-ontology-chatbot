"""페이지 후보와 용어 정합 결정을 append-only 이력으로 기록한다."""

from __future__ import annotations

import re
from pathlib import Path

from adapters.rdf import build_draft_graph, write_graph
from adapters.storage import append_jsonl, utc_now
from config import CANDIDATE_SCHEMA_VERSION, DATA_ROOT
from domain.models import EvidenceUnit, PopulationCandidate
from domain.vocabulary import ONTOLOGY_TERMS
from pipeline.validation import validate_candidate

from review.state import (
    alignment_decision_path,
    candidate_hash,
    population_decision_path,
)

IRI = re.compile(r"^(?:https?://|urn:)[^\s<>]+$")
UPSTREAM_IRIS = set(ONTOLOGY_TERMS)


def append_population_decision(
    run_root: Path,
    candidate: dict[str, object],
    decision: str,
    reviewer: str,
    reason: str = "",
) -> None:
    """현재 페이지 후보에 대한 승인 또는 거절을 새 이력 행으로 남긴다."""
    if decision not in {"accept", "reject", "amend"}:
        raise ValueError(f"Invalid population decision: {decision}")
    if decision in {"accept", "amend"} and (
        candidate.get("status") != "success"
        or not isinstance(candidate.get("coverage"), dict)
        or candidate["coverage"].get("complete") is not True
    ):
        raise ValueError("Only a complete successful candidate can be approved")
    if decision == "reject" and not reason.strip():
        raise ValueError("A rejection reason is required")
    append_jsonl(
        population_decision_path(run_root),
        {
            "schema_version": "population-review-v1",
            "unit_id": str(candidate["unit_id"]),
            "candidate_hash": candidate_hash(candidate),
            "decision": decision,
            "reviewer": reviewer.strip() or "anonymous",
            "reason": reason.strip(),
            "created_at": utc_now(),
        },
    )


def amend_population_candidate(
    run_id: str,
    run_root: Path,
    unit: EvidenceUnit,
    current: PopulationCandidate,
    amendment: dict[str, object],
    reviewer: str,
    reason: str,
) -> PopulationCandidate:
    """GUI에서 고친 JSON을 검증해 새 checkpoint와 draft TTL로 추가한다."""
    raw_entities = amendment.get("entities")
    raw_facts = amendment.get("facts")
    raw_no_fact = amendment.get("no_fact_locators")
    if not isinstance(raw_entities, list) or not all(
        isinstance(item, dict) for item in raw_entities
    ):
        raise ValueError("entities must be an array of objects")
    if not isinstance(raw_facts, list) or not all(
        isinstance(item, dict) for item in raw_facts
    ):
        raise ValueError("facts must be an array of objects")
    if not isinstance(raw_no_fact, list) or not all(
        isinstance(item, str) for item in raw_no_fact
    ):
        raise ValueError("no_fact_locators must be an array of strings")
    if len(raw_no_fact) != len(set(raw_no_fact)):
        raise ValueError("no_fact_locators contains duplicates")

    context = unit.get("context", {})
    evidence = context.get("evidence", {}) if isinstance(context, dict) else {}
    locators = [str(item) for item in context.get("locators", [])]
    facts: list[dict[str, object]] = []
    for index, raw in enumerate(raw_facts, start=1):
        locator = str(raw.get("evidence_locator", ""))
        locator_evidence = evidence.get(locator) if isinstance(evidence, dict) else None
        obj = raw.get("object")
        if not isinstance(locator_evidence, dict):
            raise TypeError(f"fact {index} uses an unknown evidence locator")
        if not isinstance(obj, dict):
            raise TypeError(f"fact {index} object must be an object")
        facts.append(
            {
                "subject": str(raw.get("subject", "")),
                "predicate": str(raw.get("predicate", "")),
                "object": {"kind": obj.get("kind"), "value": obj.get("value")},
                "evidence_locator": locator,
                # 사람이 수정할 수 없는 grounding 필드는 authoritative Markdown에서 복구한다.
                "evidence_quote": str(locator_evidence.get("quote", "")),
                "confidence": 1.0,
                "temporal_scope": None,
            }
        )

    fact_locators = {str(item["evidence_locator"]) for item in facts}
    no_fact_locators = set(raw_no_fact)
    all_locators = set(locators)
    unresolved = all_locators - fact_locators - no_fact_locators
    overlap = fact_locators & no_fact_locators
    candidate: PopulationCandidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "unit_id": str(unit["unit_id"]),
        "unit_content_hash": str(unit["content_hash"]),
        "terms": [],
        "entities": [
            {
                "entity_id": str(item.get("entity_id", "")),
                "label": str(item.get("label", "")),
                "identifier": item.get("identifier"),
                "types": list(item.get("types", []))
                if isinstance(item.get("types"), list)
                else [],
            }
            for item in raw_entities
        ],
        "facts": facts,
        "no_fact_locators": sorted(no_fact_locators),
        "coverage": {
            "target_locators": locators,
            "fact_locators": sorted(fact_locators),
            "no_fact_locators": sorted(no_fact_locators),
            "failed_locators": [],
            "unresolved_locators": sorted(unresolved | overlap),
            "complete": not unresolved and not overlap,
        },
        "generation": dict(current.get("generation", {})),
        "status": "success",
        "issues": [],
        "created_at": utc_now(),
        "model_digest": str(current.get("model_digest", "")),
        "example_set_hash": str(current.get("example_set_hash", "")),
    }
    issues = validate_candidate(unit, candidate)
    if issues:
        raise ValueError("Amended candidate is invalid: " + "; ".join(issues[:8]))

    append_jsonl(run_root / "candidates.jsonl", candidate)
    ttl_path = (
        DATA_ROOT
        / "draft"
        / run_id
        / str(unit["document_id"])
        / f"{unit['unit_id']}.ttl"
    )
    if facts:
        write_graph(
            ttl_path,
            build_draft_graph(
                run_id,
                unit,
                candidate,
                str(candidate["model_digest"]),
            ),
        )
    else:
        ttl_path.unlink(missing_ok=True)
    append_population_decision(
        run_root,
        candidate,
        "amend",
        reviewer,
        reason,
    )
    return candidate


def append_alignment_decision(
    run_root: Path,
    alignment_id: str,
    decision: str,
    reviewer: str,
    target_iri: str | None = None,
    amended_label: str | None = None,
    amended_definition: str | None = None,
) -> None:
    if decision not in {"merge", "keep_separate", "select_upstream", "amend"}:
        raise ValueError(f"Invalid alignment decision: {decision}")
    if decision in {"merge", "select_upstream"} and not target_iri:
        raise ValueError("A target IRI is required for merge decisions")
    if target_iri and not IRI.fullmatch(target_iri):
        raise ValueError("Target must be an absolute HTTP(S) or URN IRI")
    # select_upstream은 고정 contract의 IRI만 허용하고 임의 외부 IRI를 막는다.
    if decision == "select_upstream" and target_iri not in UPSTREAM_IRIS:
        raise ValueError(
            "select_upstream target is outside the locked application profile"
        )
    if decision == "amend" and not (amended_label and amended_definition):
        raise ValueError("Amended label and definition are required")

    # 동일 alignment의 재검수도 새 행으로 남기며 reader가 마지막 결정을 사용한다.
    append_jsonl(
        alignment_decision_path(run_root),
        {
            "schema_version": "alignment-review-v1",
            "alignment_id": alignment_id,
            "decision": decision,
            "target_iri": target_iri,
            "amended_label": amended_label,
            "amended_definition": amended_definition,
            "reviewer": reviewer.strip() or "anonymous",
            "created_at": utc_now(),
        },
    )
