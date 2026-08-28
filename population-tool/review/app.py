"""용어 정합, 사람 검수, RDF 검증과 발행을 한 화면에서 수행한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st
from adapters.storage import latest_by, read_json, read_jsonl, sha256_file
from config import DATA_ROOT, OLLAMA_URL
from pipeline.service import publish_run, reconcile_run, validate_population
from pipeline.validation import validate_candidate

from review.decisions import (
    amend_population_candidate,
    append_alignment_decision,
    append_population_decision,
)
from review.state import (
    alignment_decisions,
    current_population_decisions,
    population_decisions,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args, _ = parser.parse_known_args()
    return args


def _alignment_progress(
    root: Path,
) -> tuple[bool, list[dict[str, object]], dict[str, dict[str, object]], int]:
    path = root / "alignments.jsonl"
    if not path.exists():
        return False, [], {}, 0
    records = list(read_jsonl(path))
    decisions = alignment_decisions(root)
    unresolved = sum(
        item.get("status") == "needs_review"
        and str(item["alignment_id"]) not in decisions
        for item in records
    )
    return True, records, decisions, unresolved


def _candidate_panel(run_id: str, root: Path, reviewer: str) -> None:
    """페이지 원문과 사람이 검수할 Turtle을 나란히 보여준다."""
    units_path = root / "units.jsonl"
    candidates_path = root / "candidates.jsonl"
    if not units_path.exists() or not candidates_path.exists():
        st.error("먼저 generate 명령으로 ontology 초안을 생성하세요.")
        return

    units = {str(item["unit_id"]): item for item in read_jsonl(units_path)}
    latest = latest_by(read_jsonl(candidates_path), "unit_id")
    all_decisions = population_decisions(root)
    decisions = current_population_decisions(root, latest)

    records = [
        (unit_id, units[unit_id], candidate)
        for unit_id, candidate in latest.items()
        if unit_id in units
    ]
    success = sum(item[2].get("status") == "success" for item in records)
    partial = sum(item[2].get("status") == "partial" for item in records)
    failed = sum(item[2].get("status") == "failed" for item in records)
    accepted = sum(
        decisions.get(item[0], {}).get("decision") in {"accept", "amend"}
        for item in records
    )
    rejected = sum(
        decisions.get(item[0], {}).get("decision") == "reject" for item in records
    )
    pending = len(units) - accepted - rejected
    for column, (label, value) in zip(
        st.columns(6),
        (
            ("성공", success),
            ("부분", partial),
            ("실패", failed),
            ("승인", accepted),
            ("거절", rejected),
            ("미검수", pending),
        ),
    ):
        column.metric(label, value)

    selected_filter = st.selectbox(
        "표시할 결과",
        ["전체", "미검수", "승인", "거절", "사실 있음", "부분", "실패"],
    )
    if selected_filter == "미검수":
        records = [item for item in records if item[0] not in decisions]
    elif selected_filter == "승인":
        records = [
            item
            for item in records
            if decisions.get(item[0], {}).get("decision") in {"accept", "amend"}
        ]
    elif selected_filter == "거절":
        records = [
            item
            for item in records
            if decisions.get(item[0], {}).get("decision") == "reject"
        ]
    elif selected_filter == "사실 있음":
        records = [item for item in records if item[2].get("facts")]
    elif selected_filter == "부분":
        records = [item for item in records if item[2].get("status") == "partial"]
    elif selected_filter == "실패":
        records = [item for item in records if item[2].get("status") == "failed"]

    if not records:
        st.info("선택한 조건에 해당하는 결과가 없습니다.")
        return
    record_map = {item[0]: item for item in records}
    selected_id = st.selectbox(
        "근거 단위",
        list(record_map),
        format_func=lambda unit_id: (
            f"{unit_id} · {record_map[unit_id][2].get('status')} · "
            f"{decisions.get(unit_id, {}).get('decision', '미검수')}"
        ),
    )
    _, unit, candidate = record_map[selected_id]
    st.caption(
        f"PDF {unit.get('pdf_page')} · {unit.get('locator')} · "
        f"source={unit.get('source_status')}"
    )
    evidence_column, turtle_column = st.columns(2)
    with evidence_column:
        st.markdown("**페이지 원문 근거**")
        st.code(str(unit.get("text", "")), language=None, wrap_lines=True)

    if candidate.get("status") in {"failed", "partial"}:
        label = "부분 생성" if candidate.get("status") == "partial" else "생성 실패"
        st.error(f"{label} 후보입니다. 승인 전에 수정하거나 다시 생성해야 합니다.")
        for issue in candidate.get("issues", []):
            st.code(str(issue), language=None)
        if candidate.get("error"):
            st.code(str(candidate["error"]), language=None)

    if candidate.get("status") != "failed":
        generation = candidate.get("generation", {})
        extraction = (
            generation.get("extraction", {}) if isinstance(generation, dict) else {}
        )
        failed_batches = (
            int(extraction.get("failed_batch_count", 0))
            if isinstance(extraction, dict)
            else 0
        )
        if failed_batches:
            st.warning(
                f"이 페이지의 출력 묶음 {failed_batches}개가 모델 길이 제한 등으로 "
                "실패했습니다. 남은 사실만 TTL에 보존되었습니다."
            )
        ttl_path = (
            DATA_ROOT
            / "draft"
            / run_id
            / str(unit["document_id"])
            / f"{selected_id}.ttl"
        )
        with turtle_column:
            st.markdown("**페이지 ontology (Turtle)**")
            if ttl_path.exists():
                st.code(
                    ttl_path.read_text(encoding="utf-8"),
                    language="turtle",
                    wrap_lines=True,
                )
            else:
                st.info("이 페이지에는 확정적으로 추출된 fact가 없어 TTL이 없습니다.")

        with st.expander("내부 구조화 추출 보기"):
            entity_column, fact_column = st.columns(2)
            with entity_column:
                st.markdown(f"**Entities ({len(candidate.get('entities', []))})**")
                st.json(candidate.get("entities", []), expanded=True)
            with fact_column:
                st.markdown(f"**Facts ({len(candidate.get('facts', []))})**")
                st.json(candidate.get("facts", []), expanded=True)

        current_decision = decisions.get(selected_id)
        stale_decision = (
            all_decisions.get(selected_id) if current_decision is None else None
        )
        if current_decision:
            action = str(current_decision.get("decision"))
            if action in {"accept", "amend"}:
                st.success(f"현재 후보 검수 완료: {action}")
            else:
                st.error(
                    f"현재 후보 거절: {current_decision.get('reason') or '사유 없음'}"
                )
        elif stale_decision:
            st.warning("후보가 변경되어 이전 검수 결정은 더 이상 유효하지 않습니다.")
        else:
            st.warning("이 페이지는 아직 검수되지 않았습니다.")

        reason = st.text_input("검수 사유/메모", key=f"reason-{selected_id}")
        accept_column, reject_column = st.columns(2)
        if accept_column.button(
            "현재 페이지 승인",
            type="primary",
            disabled=candidate.get("status") != "success",
            key=f"accept-{selected_id}",
        ):
            issues = validate_candidate(unit, candidate)
            if issues:
                st.error("승인할 수 없습니다: " + "; ".join(issues[:8]))
            else:
                append_population_decision(root, candidate, "accept", reviewer, reason)
                st.rerun()
        if reject_column.button("현재 페이지 거절", key=f"reject-{selected_id}"):
            try:
                append_population_decision(root, candidate, "reject", reviewer, reason)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()

        editable = {
            "entities": candidate.get("entities", []),
            "facts": [
                {
                    "subject": fact.get("subject"),
                    "predicate": fact.get("predicate"),
                    "object": fact.get("object"),
                    "evidence_locator": fact.get("evidence_locator"),
                }
                for fact in candidate.get("facts", [])
                if isinstance(fact, dict)
            ],
            "no_fact_locators": candidate.get("no_fact_locators", []),
        }
        with st.expander("구조화 결과 수정 후 승인"):
            edited = st.text_area(
                "수정 가능한 JSON",
                value=json.dumps(editable, ensure_ascii=False, indent=2),
                height=520,
                key=f"amend-json-{selected_id}",
            )
            if st.button("수정본 검증 및 승인", key=f"amend-{selected_id}"):
                try:
                    payload = json.loads(edited)
                    if not isinstance(payload, dict):
                        raise TypeError("수정본은 JSON object여야 합니다")
                    amend_population_candidate(
                        run_id,
                        root,
                        unit,
                        candidate,
                        payload,
                        reviewer,
                        reason,
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.rerun()

    with st.expander("생성 provenance"):
        st.json(candidate.get("generation"))


def _reconciliation_panel(
    run_id: str,
    root: Path,
    reviewer: str,
    ollama_url: str,
) -> None:
    st.subheader("1. 용어 정합 및 검수")
    if not (root / "candidates.jsonl").exists():
        st.error("먼저 generate 명령으로 ontology 초안을 생성하세요.")
        return

    alignments_path = root / "alignments.jsonl"
    button_label = (
        "용어 정합 다시 계산" if alignments_path.exists() else "용어 정합 실행"
    )
    if st.button(
        button_label, type="primary" if not alignments_path.exists() else "secondary"
    ):
        try:
            with st.spinner("승인된 후보의 용어와 개체를 통합하고 있습니다."):
                result = reconcile_run(run_id, ollama_url)
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            st.success(
                f"정합 완료: 용어 {result['term_alignments']}개, "
                f"검수 필요 {result['needs_review']}개"
            )
            st.rerun()

    reconciled, records, decisions, unresolved = _alignment_progress(root)
    if not reconciled:
        st.info("용어 정합을 실행하면 검수 대상이 여기에 표시됩니다.")
        return
    if not records:
        st.success("정합할 신규 로컬 용어가 없습니다.")
        return

    needs_review = [item for item in records if item.get("status") == "needs_review"]
    columns = st.columns(4)
    columns[0].metric("전체 용어", len(records))
    columns[1].metric(
        "자동 처리",
        sum(item.get("status") != "needs_review" for item in records),
    )
    columns[2].metric("사람 검수 대상", len(needs_review))
    columns[3].metric("미검수", unresolved)

    if not needs_review:
        st.success("사람이 결정해야 할 용어가 없습니다.")
        return

    # 자동 처리 항목은 요약만 보여주고 모호한 alignment만 사람이 결정한다.
    for alignment in needs_review:
        alignment_id = str(alignment["alignment_id"])
        current = decisions.get(alignment_id)
        candidate = alignment["candidate"]
        label = candidate.get("label") if isinstance(candidate, dict) else alignment_id
        with st.expander(
            f"{label} · {current.get('decision') if current else '미검수'}"
        ):
            left, right = st.columns(2)
            with left:
                st.markdown("**생성 용어**")
                st.json(alignment["candidate"])
            with right:
                st.markdown("**추천 대상**")
                st.json(alignment.get("target"))
            st.markdown("**유사도 및 호환성**")
            st.json(alignment.get("score"))

            target = st.text_input(
                "Target IRI",
                value=str(alignment["target_iri"]),
                key=f"target-{alignment_id}",
            )
            amended_label = st.text_input(
                "수정 label",
                value=str(candidate.get("label", ""))
                if isinstance(candidate, dict)
                else "",
                key=f"label-{alignment_id}",
            )
            amended_definition = st.text_area(
                "수정 definition",
                value=str(candidate.get("definition", ""))
                if isinstance(candidate, dict)
                else "",
                key=f"definition-{alignment_id}",
            )
            actions = [
                ("추천 용어로 병합", "merge"),
                ("별도 용어 유지", "keep_separate"),
                ("Upstream 선택", "select_upstream"),
                ("로컬 용어 수정", "amend"),
            ]
            for column, (caption, action) in zip(st.columns(4), actions):
                if not column.button(caption, key=f"{action}-{alignment_id}"):
                    continue
                try:
                    append_alignment_decision(
                        root,
                        alignment_id,
                        action,
                        reviewer,
                        target_iri=target
                        if action in {"merge", "select_upstream"}
                        else None,
                        amended_label=amended_label if action == "amend" else None,
                        amended_definition=(
                            amended_definition if action == "amend" else None
                        ),
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()


def _validation_report(root: Path) -> dict[str, object] | None:
    path = root / "validation.json"
    if not path.exists():
        return None
    report = read_json(path)
    return report if isinstance(report, dict) else None


def _validation_is_current(root: Path, report: dict[str, object]) -> bool:
    inputs = report.get("inputs", {})
    if not isinstance(inputs, dict):
        return False
    paths = [
        root / "population-decisions.jsonl",
        root / "alignments.jsonl",
        root / "alignment-decisions.jsonl",
        root / "entity-mappings.jsonl",
    ]
    expected = {
        "units_sha256": sha256_file(root / "units.jsonl")
        if (root / "units.jsonl").exists()
        else None,
        "candidates_sha256": sha256_file(root / "candidates.jsonl")
        if (root / "candidates.jsonl").exists()
        else None,
        **{
            path.name.replace(".", "_") + "_sha256": (
                sha256_file(path) if path.exists() else None
            )
            for path in paths
        },
    }
    return all(inputs.get(key) == value for key, value in expected.items())


def _reconciliation_is_current(root: Path) -> bool:
    manifest_path = root / "manifest.json"
    candidates = root / "candidates.jsonl"
    decisions = root / "population-decisions.jsonl"
    if not manifest_path.exists() or not candidates.exists() or not decisions.exists():
        return False
    manifest = read_json(manifest_path)
    reconciliation = (
        manifest.get("reconciliation", {}) if isinstance(manifest, dict) else {}
    )
    inputs = (
        reconciliation.get("inputs", {}) if isinstance(reconciliation, dict) else {}
    )
    return bool(
        isinstance(inputs, dict)
        and inputs.get("candidates_sha256") == sha256_file(candidates)
        and inputs.get("population_decisions_sha256") == sha256_file(decisions)
    )


def _publication_panel(run_id: str, root: Path) -> None:
    st.subheader("2. RDF 검증 및 최종 발행")
    reconciled, _, _, unresolved = _alignment_progress(root)
    reconciliation_ready = bool(
        reconciled
        and (root / "entity-mappings.jsonl").exists()
        and unresolved == 0
        and _reconciliation_is_current(root)
    )
    if not reconciled:
        st.info("먼저 용어 정합을 실행하세요.")
    elif unresolved:
        st.warning(f"미검수 용어 {unresolved}개를 먼저 결정하세요.")
    elif not reconciliation_ready:
        st.warning(
            "페이지 후보 또는 검수 결과가 바뀌었습니다. 용어 통합을 다시 실행하세요."
        )

    if st.button("RDF 검증 실행", disabled=not reconciliation_ready):
        try:
            with st.spinner("근거와 draft RDF를 검증하고 있습니다."):
                report = validate_population(run_id)
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            if report["valid"]:
                st.success("RDF와 원문 근거 검증을 통과했습니다.")
            else:
                st.error(f"검증 오류 {report['counts']['issues']}개가 있습니다.")
            st.rerun()

    report = _validation_report(root)
    validation_current = bool(report and _validation_is_current(root, report))
    validation_ready = bool(report and report.get("valid") and validation_current)
    if report:
        st.json(report["counts"])
        if not validation_current:
            st.warning("초안이 검증 후 변경되었습니다. RDF 검증을 다시 실행하세요.")
        issues = report.get("issues", [])
        if issues:
            with st.expander(f"검증 오류 {len(issues)}개", expanded=True):
                for issue in issues:
                    st.code(str(issue), language=None)
        elif validation_ready:
            st.success("현재 draft는 검증을 통과했습니다.")

    publish_ready = reconciliation_ready and validation_ready
    if st.button(
        "최종 온톨로지 발행",
        type="primary",
        disabled=not publish_ready,
    ):
        try:
            with st.spinner("정합 결과를 적용해 최종 Turtle을 만들고 있습니다."):
                result = publish_run(run_id)
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            st.success(f"발행 완료: {result['path']} · {result['triples']} triples")

    published = DATA_ROOT / "published" / "hyu-course-registration-2026-2.ttl"
    if published.exists():
        st.caption(f"현재 published ontology: {published}")


def main() -> None:
    args = _args()
    root = DATA_ROOT / "runs" / args.run_id
    st.set_page_config(page_title="HYU Ontology Population", layout="wide")
    st.title("HYU Ontology Population")
    st.caption(f"run: {args.run_id}")
    reviewer = st.text_input("검수자", value="dblab")
    candidate_tab, reconciliation_tab, publication_tab = st.tabs(
        ["생성 결과", "용어 통합 및 검수", "RDF 검증 및 발행"]
    )
    with candidate_tab:
        _candidate_panel(args.run_id, root, reviewer)
    with reconciliation_tab:
        _reconciliation_panel(args.run_id, root, reviewer, args.ollama_url)
    with publication_tab:
        _publication_panel(args.run_id, root)


if __name__ == "__main__":
    main()
