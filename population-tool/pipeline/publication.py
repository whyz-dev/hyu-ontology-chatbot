"""검수된 정합 결정을 적용해 단일 published ontology를 만든다."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from adapters.rdf import (
    HYU,
    bind_namespaces,
    build_draft_graph,
    expanded_terms,
    term_iri,
    write_graph,
)
from adapters.sources import load_application_profile_graph, upstream_registry
from adapters.storage import read_jsonl, sha256_file, sha256_text, stable_json, utc_now
from domain.vocabulary import (
    APPLICATION_PROFILE_IRI,
    CANDIDATE_IRI_ROOT,
    ONTOLOGY_IRI,
    VOCAB_ROOT,
)
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD
from review.examples import example_set_hash
from review.state import alignment_decisions

from pipeline.validation import candidate_iris, validate_rdf

SCHEMA_PREDICATES = {
    RDF.type,
    RDFS.label,
    RDFS.comment,
    RDFS.domain,
    RDFS.range,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
}


def resolved_alignments(
    run_root: Path,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    alignments = list(read_jsonl(run_root / "alignments.jsonl"))
    decisions = alignment_decisions(run_root)
    mapping: dict[str, str] = {}
    amendments: dict[str, dict[str, str]] = {}
    pending = []
    # 자동 정합과 사람 결정을 하나의 candidate→canonical IRI 매핑으로 만든다.
    for alignment in alignments:
        candidate_iri = str(alignment["candidate_iri"])
        if alignment["status"] != "needs_review":
            mapping[candidate_iri] = str(alignment["target_iri"])
            continue
        decision = decisions.get(str(alignment["alignment_id"]))
        if not decision:
            pending.append(str(alignment["alignment_id"]))
            continue
        action = decision.get("decision")
        if action in {"merge", "select_upstream"}:
            mapping[candidate_iri] = str(decision["target_iri"])
        elif action == "keep_separate":
            mapping[candidate_iri] = str(alignment["distinct_iri"])
        elif action == "amend":
            target = str(alignment["distinct_iri"])
            mapping[candidate_iri] = target
            amendments[target] = {
                "label": str(decision["amended_label"]),
                "definition": str(decision["amended_definition"]),
            }
        else:
            pending.append(str(alignment["alignment_id"]))
    # 모호한 용어가 하나라도 남으면 임의로 발행하지 않는다.
    if pending:
        raise RuntimeError(f"Unreviewed term alignments remain: {len(pending)}")
    return mapping, amendments


def entity_mapping(run_root: Path) -> dict[str, str]:
    return {
        str(item["candidate_iri"]): str(item["canonical_iri"])
        for item in read_jsonl(run_root / "entity-mappings.jsonl")
    }


def _replace(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, URIRef) and str(value) in mapping:
        return URIRef(mapping[str(value)])
    return value


def _term_records(
    run_id: str,
    candidates: list[dict[str, object]],
    term_mapping: dict[str, str],
) -> dict[str, list[tuple[dict[str, object], dict[str, object]]]]:
    grouped: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(
        list
    )
    for candidate in candidates:
        unit_id = str(candidate["unit_id"])
        terms = expanded_terms(candidate)
        for term in terms:
            if term.get("iri"):
                continue
            candidate_ref = str(term_iri(run_id, unit_id, str(term["term_id"]), terms))
            target = term_mapping[candidate_ref]
            if target.startswith(VOCAB_ROOT):
                grouped[target].append((term, candidate))
    return grouped


def _resolved_reference(
    run_id: str,
    candidate: dict[str, object],
    reference: str,
    mapping: dict[str, str],
) -> URIRef:
    terms = expanded_terms(candidate)
    value = term_iri(run_id, str(candidate["unit_id"]), reference, terms)
    return URIRef(mapping.get(str(value), str(value)))


def _add_local_schema(
    graph: Graph,
    run_id: str,
    grouped: dict[str, list[tuple[dict[str, object], dict[str, object]]]],
    mapping: dict[str, str],
    amendments: dict[str, dict[str, str]],
) -> None:
    # 같은 canonical IRI로 합쳐진 여러 제안에서 대표 label/definition을 결정한다.
    for iri, records in sorted(grouped.items()):
        ref = URIRef(iri)
        kinds = Counter(str(term["kind"]) for term, _ in records)
        kind = min(kinds, key=lambda item: (-kinds[item], item))
        rdf_type = {
            "class": OWL.Class,
            "object_property": OWL.ObjectProperty,
            "datatype_property": OWL.DatatypeProperty,
        }[kind]
        labels = sorted(
            {str(term["label"]) for term, _ in records},
            key=lambda item: (len(item), item),
        )
        definitions = sorted(
            {
                str(term["definition"])
                for term, _ in records
                if str(term.get("definition", "")).strip()
            },
            key=lambda item: (len(item), item),
        )
        amendment = amendments.get(iri)
        label = amendment["label"] if amendment else labels[0]
        definition = (
            amendment["definition"]
            if amendment
            else (definitions[0] if definitions else label)
        )
        graph.add((ref, RDF.type, rdf_type))
        graph.add((ref, RDFS.label, Literal(label, lang="ko")))
        graph.add((ref, RDFS.comment, Literal(definition, lang="ko")))
        for term, candidate in records:
            parent = term.get("upstream_parent_iri")
            if parent:
                relation = RDFS.subClassOf if kind == "class" else RDFS.subPropertyOf
                graph.add((ref, relation, URIRef(str(parent))))
            if kind == "class":
                continue
            for domain in term.get("domains", []):
                graph.add(
                    (
                        ref,
                        RDFS.domain,
                        _resolved_reference(run_id, candidate, str(domain), mapping),
                    )
                )
            for range_value in term.get("ranges", []):
                graph.add(
                    (
                        ref,
                        RDFS.range,
                        _resolved_reference(
                            run_id, candidate, str(range_value), mapping
                        ),
                    )
                )


def publish_ontology(
    run_id: str,
    run_root: Path,
    output_path: Path,
    units: dict[str, dict[str, object]],
    candidates_by_unit: dict[str, dict[str, object]],
) -> Graph:
    validation_report = run_root / "validation.json"
    if not validation_report.exists():
        raise RuntimeError("Run validation is missing; validate it in the review GUI")
    import json

    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise RuntimeError("Run validation failed; ontology was not published")
    inputs = report.get("inputs", {})
    mutable_paths = [
        run_root / "population-decisions.jsonl",
        run_root / "alignments.jsonl",
        run_root / "alignment-decisions.jsonl",
        run_root / "entity-mappings.jsonl",
    ]
    current_inputs = {
        "units_sha256": sha256_file(run_root / "units.jsonl"),
        "candidates_sha256": sha256_file(run_root / "candidates.jsonl"),
        **{
            path.name.replace(".", "_") + "_sha256": (
                sha256_file(path) if path.exists() else None
            )
            for path in mutable_paths
        },
    }
    if not isinstance(inputs, dict) or any(
        inputs.get(key) != value for key, value in current_inputs.items()
    ):
        raise RuntimeError(
            "Run inputs changed after validation; validate again in the review GUI"
        )
    term_mapping, amendments = resolved_alignments(run_root)
    all_mapping = {**term_mapping, **entity_mapping(run_root)}
    candidates = list(candidates_by_unit.values())

    graph = Graph()
    bind_namespaces(graph)
    profile_ref = URIRef(APPLICATION_PROFILE_IRI)
    for subject, predicate, obj in load_application_profile_graph():
        if subject != profile_ref:
            graph.add((subject, predicate, obj))
    # 검수 전 raw TTL을 복사하지 않고 승인된 최신 후보에서 graph를 다시 만든다.
    for unit_id, candidate in sorted(candidates_by_unit.items()):
        if not candidate.get("facts"):
            continue
        source = build_draft_graph(
            run_id,
            units[unit_id],
            candidate,
            str(candidate.get("model_digest", "")),
        )
        for subject, predicate, obj in source:
            if (
                str(subject).startswith(CANDIDATE_IRI_ROOT)
                and predicate in SCHEMA_PREDICATES
            ):
                continue
            graph.add(
                (
                    _replace(subject, all_mapping),
                    _replace(predicate, all_mapping),
                    _replace(obj, all_mapping),
                )
            )

    # canonical IRI 기준으로 로컬 schema를 한 번만 다시 선언한다.
    grouped = _term_records(run_id, candidates, term_mapping)
    _add_local_schema(graph, run_id, grouped, term_mapping, amendments)
    ontology = URIRef(ONTOLOGY_IRI)
    graph.add((ontology, RDF.type, OWL.Ontology))
    graph.add(
        (
            ontology,
            DCTERMS.title,
            Literal("한양대학교 2026학년도 2학기 수강신청 온톨로지", lang="ko"),
        )
    )
    graph.add((ontology, DCTERMS.conformsTo, URIRef(APPLICATION_PROFILE_IRI)))
    graph.add((ontology, DCTERMS.issued, Literal(utc_now(), datatype=XSD.dateTime)))
    graph.add(
        (
            ontology,
            HYU.exampleSetHash,
            Literal(example_set_hash(), datatype=XSD.string),
        )
    )
    decision_hashes = {
        path.name: sha256_file(path)
        for path in (
            run_root / "population-decisions.jsonl",
            run_root / "alignment-decisions.jsonl",
        )
        if path.exists()
    }
    if decision_hashes:
        graph.add(
            (
                ontology,
                HYU.reviewDecisionHash,
                Literal(
                    sha256_text(stable_json(decision_hashes)),
                    datatype=XSD.string,
                ),
            )
        )
    for source in upstream_registry():
        graph.add((ontology, DCTERMS.conformsTo, URIRef(str(source["specification"]))))
    unresolved = candidate_iris(graph)
    if unresolved:
        raise RuntimeError(
            f"Published graph still contains {len(unresolved)} candidate IRIs"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # published에는 고정 profile과 최종 ontology 한 파일만 남긴다.
    allowed_entries = {output_path, output_path.parent / "profile"}
    other_files = [
        path for path in output_path.parent.iterdir() if path not in allowed_entries
    ]
    if other_files:
        raise RuntimeError(
            "published directory must contain only profile/ and the final TTL: "
            + ", ".join(path.name for path in other_files)
        )
    write_graph(output_path, graph)
    valid, rdf_report = validate_rdf(output_path)
    if not valid:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Published ontology failed RDF validation: {rdf_report}")
    return graph
