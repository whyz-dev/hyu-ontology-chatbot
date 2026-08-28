"""구조화된 추출 후보를 provenance가 포함된 draft RDF로 변환한다."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from domain.models import EvidenceUnit, PopulationCandidate
from domain.vocabulary import (
    ACTIVITY_IRI_ROOT,
    ASSERTION_IRI_ROOT,
    CANDIDATE_IRI_ROOT,
    DOCUMENT_IRI_ROOT,
    UNIT_IRI_ROOT,
    VOCAB_ROOT,
)
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, RDF, RDFS, SKOS, XSD

from adapters.storage import sha256_text, stable_json, write_text

HYU = Namespace(VOCAB_ROOT)
TIME = Namespace("http://www.w3.org/2006/time#")
SCHEMA = Namespace("https://schema.org/")


def bind_namespaces(graph: Graph) -> None:
    graph.bind("hyu", HYU)
    graph.bind("dcterms", DCTERMS)
    graph.bind("owl", OWL)
    graph.bind("prov", PROV)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("schema", SCHEMA)
    graph.bind("skos", SKOS)
    graph.bind("time", TIME)
    graph.bind("xsd", XSD)


def _safe(value: object) -> str:
    result = re.sub(r"[^0-9A-Za-z가-힣._~-]+", "-", str(value)).strip("-")
    return result[:100] or sha256_text(str(value))[:16]


def implicit_term_id(label: str) -> str:
    return f"implicit-{sha256_text(label)[:12]}"


def expanded_terms(candidate: dict[str, object]) -> list[dict[str, object]]:
    terms = [
        dict(item) for item in candidate.get("terms", []) if isinstance(item, dict)
    ]
    known_labels = {str(item.get("label")) for item in terms}
    referenced = set()
    for entity in candidate.get("entities", []):
        if isinstance(entity, dict):
            referenced.update(
                str(item) for item in entity.get("types", []) if str(item)
            )
    for term in terms:
        referenced.update(str(item) for item in term.get("domains", []) if str(item))
        referenced.update(str(item) for item in term.get("ranges", []) if str(item))
    # domain/range나 entity type에서만 등장한 로컬 class도 RDF 선언 대상으로 보완한다.
    for label in sorted(referenced - known_labels):
        if label.startswith(("http://", "https://", "urn:")):
            continue
        terms.append(
            {
                "term_id": implicit_term_id(label),
                "kind": "class",
                "iri": None,
                "label": label,
                "definition": f"근거와 생성된 schema에서 참조된 {label} 개념.",
                "domains": [],
                "ranges": [],
                "upstream_parent_iri": None,
                "implicit": True,
            }
        )
    return terms


def candidate_term_iri(run_id: str, unit_id: str, term_id: str) -> URIRef:
    # 자유 생성 용어는 정합 전까지 unit-local IRI로 격리한다.
    return URIRef(
        f"{CANDIDATE_IRI_ROOT}{_safe(run_id)}:{_safe(unit_id)}:term:{_safe(term_id)}"
    )


def candidate_entity_iri(run_id: str, unit_id: str, entity_id: str) -> URIRef:
    return URIRef(
        f"{CANDIDATE_IRI_ROOT}{_safe(run_id)}:{_safe(unit_id)}:entity:{_safe(entity_id)}"
    )


def term_iri(
    run_id: str,
    unit_id: str,
    reference: str,
    terms: list[dict[str, object]],
) -> URIRef:
    if reference.startswith(("http://", "https://", "urn:")):
        return URIRef(reference)
    by_id = {str(item.get("term_id")): item for item in terms}
    by_label = {str(item.get("label")): item for item in terms}
    term = by_id.get(reference) or by_label.get(reference)
    if term and term.get("iri"):
        return URIRef(str(term["iri"]))
    term_id = str(term.get("term_id")) if term else implicit_term_id(reference)
    return candidate_term_iri(run_id, unit_id, term_id)


def _literal(obj: dict[str, object]) -> Literal:
    kind = obj["kind"]
    value = obj.get("value")
    if kind == "integer":
        return Literal(int(value), datatype=XSD.integer)
    if kind == "decimal":
        return Literal(Decimal(str(value)), datatype=XSD.decimal)
    if kind == "boolean":
        return Literal(bool(value), datatype=XSD.boolean)
    if kind == "date":
        return Literal(str(value), datatype=XSD.date)
    if kind == "datetime":
        return Literal(str(value), datatype=XSD.dateTimeStamp)
    return Literal(str(value), lang="ko")


def build_draft_graph(
    run_id: str,
    unit: EvidenceUnit,
    candidate: PopulationCandidate,
    model_digest: str,
) -> Graph:
    graph = Graph()
    bind_namespaces(graph)
    unit_id = str(unit["unit_id"])
    terms = expanded_terms(candidate)
    # 먼저 원문 단위와 생성 활동을 기록해 모든 assertion의 출처를 추적할 수 있게 한다.
    unit_ref = URIRef(f"{UNIT_IRI_ROOT}{_safe(unit_id)}")
    document_ref = URIRef(f"{DOCUMENT_IRI_ROOT}{_safe(unit['document_id'])}")
    activity_ref = URIRef(f"{ACTIVITY_IRI_ROOT}{_safe(run_id)}:{_safe(unit_id)}")
    graph.add((unit_ref, RDF.type, HYU.EvidenceUnit))
    graph.add((unit_ref, DCTERMS.isPartOf, document_ref))
    graph.add(
        (unit_ref, HYU.locator, Literal(str(unit["locator"]), datatype=XSD.string))
    )
    graph.add(
        (
            unit_ref,
            HYU.contentHash,
            Literal(str(unit["content_hash"]), datatype=XSD.string),
        )
    )
    graph.add(
        (
            unit_ref,
            HYU.sourceStatus,
            Literal(str(unit["source_status"]), datatype=XSD.string),
        )
    )
    graph.add(
        (
            unit_ref,
            HYU.sourceHash,
            Literal(str(unit["source_hash"]), datatype=XSD.string),
        )
    )
    for issue in unit.get("source_issues", []):
        graph.add((unit_ref, HYU.sourceIssue, Literal(str(issue), datatype=XSD.string)))
    graph.add((unit_ref, SCHEMA.text, Literal(str(unit["text"]), lang="ko")))
    if unit.get("pdf_page") is not None:
        graph.add(
            (
                unit_ref,
                HYU.pdfPage,
                Literal(int(unit["pdf_page"]), datatype=XSD.integer),
            )
        )
    if unit.get("printed_page") is not None:
        graph.add(
            (
                unit_ref,
                HYU.printedPage,
                Literal(int(unit["printed_page"]), datatype=XSD.integer),
            )
        )
    graph.add((document_ref, RDF.type, PROV.Entity))
    graph.add((document_ref, DCTERMS.identifier, Literal(str(unit["document_id"]))))
    graph.add((activity_ref, RDF.type, PROV.Activity))
    graph.add(
        (activity_ref, HYU.modelDigest, Literal(model_digest, datatype=XSD.string))
    )
    graph.add(
        (
            activity_ref,
            HYU.modelTag,
            Literal(
                str(candidate["generation"]["extraction"]["model"]), datatype=XSD.string
            ),
        )
    )
    graph.add(
        (
            activity_ref,
            HYU.promptHash,
            Literal(str(candidate["generation"]["prompt_hash"]), datatype=XSD.string),
        )
    )
    if candidate.get("example_set_hash"):
        graph.add(
            (
                activity_ref,
                HYU.exampleSetHash,
                Literal(str(candidate["example_set_hash"]), datatype=XSD.string),
            )
        )
    if candidate.get("created_at"):
        graph.add(
            (
                activity_ref,
                PROV.generatedAtTime,
                Literal(str(candidate["created_at"]), datatype=XSD.dateTime),
            )
        )

    # profile/upstream에 없는 schema 후보만 draft graph에 선언한다.
    for term in terms:
        iri = term_iri(run_id, unit_id, str(term["term_id"]), terms)
        if term.get("iri"):
            continue
        rdf_type = {
            "class": OWL.Class,
            "object_property": OWL.ObjectProperty,
            "datatype_property": OWL.DatatypeProperty,
        }[str(term["kind"])]
        graph.add((iri, RDF.type, rdf_type))
        graph.add((iri, RDFS.label, Literal(str(term["label"]), lang="ko")))
        graph.add((iri, RDFS.comment, Literal(str(term["definition"]), lang="ko")))
        parent = term.get("upstream_parent_iri")
        if parent:
            relation = (
                RDFS.subClassOf if term["kind"] == "class" else RDFS.subPropertyOf
            )
            graph.add((iri, relation, URIRef(str(parent))))
        for domain in term.get("domains", []):
            graph.add((iri, RDFS.domain, term_iri(run_id, unit_id, str(domain), terms)))
        for range_value in term.get("ranges", []):
            graph.add(
                (iri, RDFS.range, term_iri(run_id, unit_id, str(range_value), terms))
            )

    entities = {
        str(item["entity_id"]): item
        for item in candidate.get("entities", [])
        if isinstance(item, dict)
    }
    entity_refs = {
        entity_id: candidate_entity_iri(run_id, unit_id, entity_id)
        for entity_id in entities
    }
    for entity_id, entity in entities.items():
        ref = entity_refs[entity_id]
        graph.add((ref, SKOS.prefLabel, Literal(str(entity["label"]), lang="ko")))
        for type_name in entity.get("types", []):
            graph.add((ref, RDF.type, term_iri(run_id, unit_id, str(type_name), terms)))
        if entity.get("identifier"):
            graph.add((ref, DCTERMS.identifier, Literal(str(entity["identifier"]))))

    # 질의용 direct triple과 감사용 rdf:Statement assertion을 함께 저장한다.
    for fact in candidate.get("facts", []):
        subject = entity_refs[str(fact["subject"])]
        predicate = term_iri(run_id, unit_id, str(fact["predicate"]), terms)
        obj = fact["object"]
        object_value = (
            entity_refs[str(obj["value"])] if obj["kind"] == "entity" else _literal(obj)
        )
        graph.add((subject, predicate, object_value))
        # 사실을 삭제·수정해도 뒤 assertion의 IRI가 연쇄적으로 바뀌지 않게 한다.
        assertion_id = sha256_text(stable_json([run_id, unit_id, fact]))[:24]
        assertion = URIRef(f"{ASSERTION_IRI_ROOT}{assertion_id}")
        graph.add((assertion, RDF.type, HYU.Assertion))
        graph.add((assertion, RDF.subject, subject))
        graph.add((assertion, RDF.predicate, predicate))
        graph.add((assertion, RDF.object, object_value))
        graph.add((assertion, PROV.wasDerivedFrom, unit_ref))
        graph.add((assertion, PROV.wasGeneratedBy, activity_ref))
        graph.add(
            (
                assertion,
                HYU.evidenceLocator,
                Literal(str(fact["evidence_locator"]), datatype=XSD.string),
            )
        )
        graph.add(
            (
                assertion,
                HYU.evidenceQuote,
                Literal(str(fact["evidence_quote"]), datatype=XSD.string),
            )
        )
        graph.add(
            (
                assertion,
                HYU.confidence,
                Literal(Decimal(str(fact["confidence"])), datatype=XSD.decimal),
            )
        )
    return graph


def write_graph(path: Path, graph: Graph) -> None:
    serialized = graph.serialize(format="turtle")
    write_text(path, str(serialized))
