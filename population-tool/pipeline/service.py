"""CLI 단계별 workflow와 checkpoint/resume 동작을 조정한다."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from adapters.ollama import OllamaClient
from adapters.rdf import build_draft_graph, write_graph
from adapters.sources import (
    fetch_upstream,
    load_upstream_graph,
)
from adapters.storage import (
    append_jsonl,
    latest_by,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)
from config import (
    APPLICATION_PROFILE_PATH,
    CANDIDATE_SCHEMA_VERSION,
    CANONICAL_DOCUMENT,
    DATA_ROOT,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    OLLAMA_URL,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from domain.vocabulary import (
    APPLICATION_PROFILE_IRI,
    APPLICATION_PROFILE_VERSION,
)
from review.examples import example_set_hash, load_examples
from review.state import reviewed_population_candidates

from pipeline.extraction import extract_unit
from pipeline.publication import publish_ontology
from pipeline.reconciliation import reconcile_terms
from pipeline.segmentation import prepare_units
from pipeline.validation import validate_candidate, validate_run

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def run_root(run_id: str) -> Path:
    # run ID가 경로로 해석되지 않게 허용 문자를 제한한다.
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run ID must contain only letters, digits, dot, underscore and hyphen"
        )
    return DATA_ROOT / "runs" / run_id


def draft_root(run_id: str) -> Path:
    return DATA_ROOT / "draft" / run_id


def prepare_run(
    run_id: str, canonical_path: Path = CANONICAL_DOCUMENT
) -> dict[str, object]:
    root = run_root(run_id)
    units_path = root / "units.jsonl"
    units = prepare_units(canonical_path, units_path)
    source_manifest = canonical_path.parent / "manifest.json"
    manifest = {
        "schema_version": "population-run-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "input": {
            "path": canonical_path.as_posix(),
            "sha256": sha256_file(canonical_path),
            "schema_version": SCHEMA_VERSION,
            "manifest_path": source_manifest.as_posix(),
            "manifest_sha256": sha256_file(source_manifest),
        },
        "units": {
            "path": units_path.as_posix(),
            "sha256": sha256_file(units_path),
            "count": len(units),
            "kinds": dict(Counter(str(item["kind"]) for item in units)),
            "source_statuses": dict(
                Counter(str(item["source_status"]) for item in units)
            ),
        },
        "models": {
            "generation": GENERATION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "application_profile": {
            "iri": APPLICATION_PROFILE_IRI,
            "version": APPLICATION_PROFILE_VERSION,
            "path": APPLICATION_PROFILE_PATH.as_posix(),
            "sha256": sha256_file(APPLICATION_PROFILE_PATH),
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _completed_units(
    path: Path,
    *,
    example_hash: str,
    model_digest: str,
) -> set[str]:
    if not path.exists():
        return set()
    # append-only checkpoint에서는 각 unit의 최신 결과만 본다. 모든 locator를
    # 분류한 partial 후보는 초안 생성이 끝난 것이므로, 의미 검수 경고가 있더라도
    # resume에서 다시 LLM을 호출하지 않는다. failed/incomplete만 재시도한다.
    latest = latest_by(read_jsonl(path), "unit_id")
    return {
        str(item["unit_id"])
        for item in latest.values()
        if item.get("status") in {"success", "partial"}
        and isinstance(item.get("coverage"), dict)
        and item["coverage"].get("complete") is True
        and item.get("example_set_hash") == example_hash
        and item.get("model_digest") == model_digest
        and isinstance(item.get("generation"), dict)
        and item["generation"].get("prompt_version") == PROMPT_VERSION
    }


def _reviewed_run_candidates(
    root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """현재 페이지 79개가 모두 완전하고 승인됐는지 한 경로에서 확인한다."""
    units_path = root / "units.jsonl"
    candidates_path = root / "candidates.jsonl"
    if not units_path.exists() or not candidates_path.exists():
        raise RuntimeError("Prepared units or candidate checkpoint is missing")
    units = latest_by(read_jsonl(units_path), "unit_id")
    candidates = latest_by(read_jsonl(candidates_path), "unit_id")
    reviewed, issues = reviewed_population_candidates(root, units, candidates)
    issues.extend(
        f"{unit_id}: {issue}"
        for unit_id, candidate in reviewed.items()
        for issue in validate_candidate(units[unit_id], candidate)
    )
    if issues:
        preview = "; ".join(issues[:8])
        remainder = f"; ... ({len(issues)} total)" if len(issues) > 8 else ""
        raise RuntimeError(
            f"Population review is incomplete or stale: {preview}{remainder}"
        )
    return units, reviewed


def populate_run(
    run_id: str,
    resume: bool,
    limit: int | None = None,
    ollama_url: str = OLLAMA_URL,
    model: str = GENERATION_MODEL,
) -> dict[str, int]:
    root = run_root(run_id)
    units_path = root / "units.jsonl"
    if not units_path.exists():
        raise RuntimeError("Prepared units are missing; run generate again")
    # 예제는 사람과 함께 확정한 published profile 자산을 그대로 사용한다.
    examples = load_examples()
    examples_hash = example_set_hash()
    client = OllamaClient(ollama_url)
    models = client.models()
    if model not in models:
        raise RuntimeError(f"Ollama model is not installed: {model}")
    graph = load_upstream_graph()
    candidates_path = root / "candidates.jsonl"
    completed_units = (
        _completed_units(
            candidates_path,
            example_hash=examples_hash,
            model_digest=models[model],
        )
        if resume
        else set()
    )
    if candidates_path.exists() and not resume:
        raise RuntimeError(
            "Candidate checkpoint already exists; use --resume or a new run ID"
        )
    units = list(read_jsonl(units_path))
    if limit is not None:
        units = units[:limit]
    counts = Counter(skipped=0, success=0, partial=0, failed=0)
    for index, unit in enumerate(units, start=1):
        unit_id = str(unit["unit_id"])
        ttl_path = draft_root(run_id) / str(unit["document_id"]) / f"{unit_id}.ttl"
        if unit_id in completed_units:
            counts["skipped"] += 1
            continue
        print(f"[{index}/{len(units)}] {unit_id}", flush=True)
        last_issues: list[str] = []
        last_error: str | None = None
        candidate = None
        try:
            proposed = extract_unit(client, model, unit, graph, examples)
            last_issues = validate_candidate(unit, proposed)
            candidate = proposed
        except (RuntimeError, ValueError) as error:
            last_error = str(error)
        if candidate is None:
            # 재실행 중 현재 시도가 실패했는데 이전 성공 TTL이 남아 있으면 GUI가
            # 최신 checkpoint와 다른 ontology를 보여주게 된다.
            ttl_path.unlink(missing_ok=True)
            append_jsonl(
                candidates_path,
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "unit_id": unit_id,
                    "unit_content_hash": unit["content_hash"],
                    "status": "failed",
                    "issues": last_issues,
                    "error": last_error,
                    "created_at": utc_now(),
                },
            )
            counts["failed"] += 1
            continue
        coverage_complete = bool(candidate.get("coverage", {}).get("complete"))
        candidate["status"] = (
            "success" if coverage_complete and not last_issues else "partial"
        )
        candidate["issues"] = last_issues
        candidate["created_at"] = utc_now()
        candidate["model_digest"] = models[model]
        candidate["example_set_hash"] = examples_hash
        candidate["generation"]["prompt_version"] = PROMPT_VERSION
        # JSONL은 내부 resume checkpoint다. 사람이 검수할 주 산출물은 page TTL이다.
        append_jsonl(candidates_path, candidate)
        if candidate.get("facts"):
            # partial TTL도 사람이 누락·오류를 확인할 수 있는 draft로 저장한다.
            # reconcile/publish는 승인된 success 후보에서 graph를 다시 만들기 때문에
            # 이 파일이 그대로 최종 ontology에 섞이지 않는다.
            write_graph(
                ttl_path,
                build_draft_graph(run_id, unit, candidate, models[model]),
            )
        else:
            # 같은 run을 resume했을 때 남아 있는 예전 TTL이 최신 partial/빈 후보를
            # 가장하지 못하게 한다.
            ttl_path.unlink(missing_ok=True)
        counts[str(candidate["status"])] += 1
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["population"] = {
        "updated_at": utc_now(),
        "model": model,
        "model_digest": models[model],
        "example_set_hash": examples_hash,
        "prompt_version": PROMPT_VERSION,
        "counts": dict(counts),
    }
    write_json(manifest_path, manifest)
    return dict(counts)


def generate_run(
    run_id: str,
    resume: bool,
    canonical_path: Path = CANONICAL_DOCUMENT,
    limit: int | None = None,
    ollama_url: str = OLLAMA_URL,
) -> dict[str, object]:
    """Upstream 확보, 문서 분해, draft 생성을 한 명령에서 수행한다."""
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical document does not exist: {canonical_path}")

    root = run_root(run_id)
    manifest_path = root / "manifest.json"
    units_path = root / "units.jsonl"
    candidates_path = root / "candidates.jsonl"
    if root.exists() and any(root.iterdir()) and not resume:
        raise RuntimeError(
            "Run output already exists; use --resume or choose a new run ID"
        )
    if resume and candidates_path.exists() and not units_path.exists():
        raise RuntimeError(
            "Candidate checkpoint exists without prepared evidence units"
        )

    # 캐시가 유효하면 다운로드하지 않고, 없거나 checksum이 다르면 다시 받는다.
    upstream = fetch_upstream()
    if resume and units_path.exists():
        if not manifest_path.exists():
            raise RuntimeError("Prepared units exist without a run manifest")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError("Run manifest is not a JSON object")
        recorded_input = manifest.get("input", {})
        source_manifest = canonical_path.parent / "manifest.json"
        if (
            not isinstance(recorded_input, dict)
            or recorded_input.get("sha256") != sha256_file(canonical_path)
            or recorded_input.get("manifest_sha256") != sha256_file(source_manifest)
        ):
            raise RuntimeError(
                "Canonical document or its manifest changed after preparation; "
                "use a new run ID"
            )
        recorded_profile = manifest.get("application_profile", {})
        if not isinstance(recorded_profile, dict) or recorded_profile.get(
            "sha256"
        ) != sha256_file(APPLICATION_PROFILE_PATH):
            raise RuntimeError(
                "Application profile changed after preparation; use a new run ID"
            )
        prepared = {
            "reused": True,
            "units": manifest.get("units", {}),
        }
    else:
        manifest = prepare_run(run_id, canonical_path)
        prepared = {
            "reused": False,
            "units": manifest["units"],
        }

    population = populate_run(
        run_id,
        resume=resume,
        limit=limit,
        ollama_url=ollama_url,
    )
    return {
        "run_id": run_id,
        "upstream": [
            {
                "name": item["name"],
                "status": item["status"],
                "path": str(item["path"]),
            }
            for item in upstream
        ],
        "preparation": prepared,
        "population": population,
    }


def reconcile_run(
    run_id: str,
    ollama_url: str = OLLAMA_URL,
) -> dict[str, int]:
    root = run_root(run_id)
    candidates_path = root / "candidates.jsonl"
    if not candidates_path.exists():
        raise RuntimeError("Candidate checkpoint is missing; run generate first")
    _, reviewed = _reviewed_run_candidates(root)
    client = OllamaClient(ollama_url)
    models = client.models()
    alignments, entities = reconcile_terms(
        run_id,
        list(reviewed.values()),
        load_upstream_graph(),
        client,
        root,
    )
    result = {
        "term_alignments": len(alignments),
        "needs_review": sum(item["status"] == "needs_review" for item in alignments),
        "entity_mappings": len(entities),
    }
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["reconciliation"] = {
        "updated_at": utc_now(),
        "model": EMBEDDING_MODEL if EMBEDDING_MODEL in models else None,
        "model_digest": models.get(EMBEDDING_MODEL),
        "inputs": {
            "candidates_sha256": sha256_file(candidates_path),
            "population_decisions_sha256": sha256_file(
                root / "population-decisions.jsonl"
            ),
        },
        "counts": result,
    }
    write_json(manifest_path, manifest)
    return result


def validate_population(run_id: str) -> dict[str, object]:
    root = run_root(run_id)
    required = [
        root / "units.jsonl",
        root / "candidates.jsonl",
        root / "population-decisions.jsonl",
        root / "alignments.jsonl",
        root / "entity-mappings.jsonl",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing run files: {', '.join(missing)}")
    manifest = read_json(root / "manifest.json")
    reconciliation = (
        manifest.get("reconciliation", {}) if isinstance(manifest, dict) else {}
    )
    reconciliation_inputs = (
        reconciliation.get("inputs", {}) if isinstance(reconciliation, dict) else {}
    )
    if not isinstance(reconciliation_inputs, dict) or (
        reconciliation_inputs.get("candidates_sha256")
        != sha256_file(root / "candidates.jsonl")
        or reconciliation_inputs.get("population_decisions_sha256")
        != sha256_file(root / "population-decisions.jsonl")
    ):
        raise RuntimeError(
            "Population candidates or review decisions changed; run reconciliation again"
        )
    _, reviewed = _reviewed_run_candidates(root)
    mutable_inputs = [
        root / "population-decisions.jsonl",
        root / "alignments.jsonl",
        root / "alignment-decisions.jsonl",
        root / "entity-mappings.jsonl",
    ]
    return validate_run(
        root / "units.jsonl",
        root / "candidates.jsonl",
        draft_root(run_id),
        root / "validation.json",
        run_id=run_id,
        reviewed_candidates=reviewed,
        mutable_inputs=mutable_inputs,
    )


def publish_run(run_id: str) -> dict[str, object]:
    root = run_root(run_id)
    # 정합과 검증 산출물이 모두 있어야 published 영역을 변경할 수 있다.
    for name in ("alignments.jsonl", "entity-mappings.jsonl", "validation.json"):
        if not (root / name).exists():
            raise RuntimeError(
                f"Missing {name}; complete reconciliation and validation in review"
            )
    units, reviewed = _reviewed_run_candidates(root)
    output = DATA_ROOT / "published" / "hyu-course-registration-2026-2.ttl"
    graph = publish_ontology(run_id, root, output, units, reviewed)
    result = {
        "path": output.as_posix(),
        "sha256": sha256_file(output),
        "triples": len(graph),
    }
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["publication"] = {"created_at": utc_now(), **result}
    write_json(manifest_path, manifest)
    return result
