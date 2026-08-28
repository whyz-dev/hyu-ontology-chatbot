"""잠금된 외부 ontology와 HYU application profile을 읽는다."""

from __future__ import annotations

import urllib.error
import urllib.request
from functools import cache
from pathlib import Path

from config import APPLICATION_PROFILE_PATH, DATA_ROOT, UPSTREAM_LOCK_PATH
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from adapters.storage import read_json, sha256_file


def upstream_registry() -> list[dict[str, str]]:
    # 외부 ontology의 URL과 checksum은 published profile의 lock 파일만 신뢰한다.
    raw = read_json(UPSTREAM_LOCK_PATH)
    if not isinstance(raw, dict) or raw.get("schema_version") != "upstream-lock-v1":
        raise ValueError(f"Invalid upstream lock: {UPSTREAM_LOCK_PATH}")
    sources = raw.get("sources")
    if not isinstance(sources, list):
        raise TypeError(f"Invalid upstream source list: {UPSTREAM_LOCK_PATH}")
    return [dict(item) for item in sources if isinstance(item, dict)]


def upstream_path(source: dict[str, str], data_root: Path = DATA_ROOT) -> Path:
    return data_root / "upstream" / source["path"]


def check_upstream(data_root: Path = DATA_ROOT) -> list[dict[str, object]]:
    results = []
    for source in upstream_registry():
        path = upstream_path(source, data_root)
        actual = sha256_file(path) if path.exists() else None
        results.append(
            {
                "name": source["name"],
                "path": path.as_posix(),
                "exists": path.exists(),
                "expected_sha256": source["sha256"],
                "actual_sha256": actual,
                "valid": actual == source["sha256"],
            }
        )
    return results


def fetch_upstream(data_root: Path = DATA_ROOT) -> list[dict[str, object]]:
    results = []
    for source in upstream_registry():
        path = upstream_path(source, data_root)
        if path.exists() and sha256_file(path) == source["sha256"]:
            results.append({"name": source["name"], "status": "cached", "path": path})
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # 다운로드가 중단돼도 정상 snapshot을 덮지 않도록 partial 파일을 사용한다.
        temporary = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(
            source["url"],
            headers={
                "Accept": "text/turtle, application/rdf+xml;q=0.9",
                "User-Agent": "hyu-chatbot-population-tool/0.1 (+local research)",
            },
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        except urllib.error.URLError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {source['name']}: {error}"
            ) from error
        # 원격 내용이 바뀌었으면 조용히 수용하지 않고 lock 갱신을 요구한다.
        actual = sha256_file(temporary)
        if actual != source["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {source['name']}: expected {source['sha256']}, got {actual}"
            )
        temporary.replace(path)
        results.append({"name": source["name"], "status": "downloaded", "path": path})
    return results


def load_upstream_graph(data_root: Path = DATA_ROOT) -> Graph:
    invalid = [item for item in check_upstream(data_root) if not item["valid"]]
    if invalid:
        names = ", ".join(str(item["name"]) for item in invalid)
        raise RuntimeError(f"Missing or invalid upstream snapshots: {names}")
    graph = Graph()
    for source in upstream_registry():
        graph.parse(upstream_path(source, data_root), format=source["format"])
    return graph


@cache
def load_application_profile_graph() -> Graph:
    # profile은 한 실행 중 불변이므로 반복 parsing하지 않는다.
    graph = Graph()
    graph.parse(APPLICATION_PROFILE_PATH, format="turtle")
    return graph


def _literal(graph: Graph, subject: URIRef, predicates: tuple[URIRef, ...]) -> str:
    values = []
    for predicate in predicates:
        for value in graph.objects(subject, predicate):
            if getattr(value, "language", None) in {None, "ko", "en"}:
                values.append(str(value))
    return min(values, key=len) if values else ""


def ontology_term(graph: Graph, iri: str) -> dict[str, object]:
    subject = URIRef(iri)
    types = {str(value) for value in graph.objects(subject, RDF.type)}
    if str(OWL.Class) in types or str(RDFS.Class) in types:
        kind = "class"
    elif str(OWL.DatatypeProperty) in types:
        kind = "datatype_property"
    else:
        kind = "object_property"
    return {
        "iri": iri,
        "kind": kind,
        "label": _literal(graph, subject, (SKOS.prefLabel, RDFS.label))
        or iri.rsplit("/", 1)[-1],
        "definition": _literal(
            graph,
            subject,
            (SKOS.definition, RDFS.comment),
        ),
        "domains": sorted(str(value) for value in graph.objects(subject, RDFS.domain)),
        "ranges": sorted(str(value) for value in graph.objects(subject, RDFS.range)),
        "parents": sorted(
            str(value)
            for predicate in (RDFS.subClassOf, RDFS.subPropertyOf)
            for value in graph.objects(subject, predicate)
            if isinstance(value, URIRef)
        ),
    }


@cache
def application_profile_terms() -> tuple[dict[str, object], ...]:
    graph = load_application_profile_graph()
    term_types = {OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty}
    subjects = {
        subject
        for term_type in term_types
        for subject in graph.subjects(RDF.type, term_type)
        if isinstance(subject, URIRef)
    }
    return tuple(ontology_term(graph, str(subject)) for subject in sorted(subjects))


@cache
def application_profile_iris() -> frozenset[str]:
    return frozenset(str(item["iri"]) for item in application_profile_terms())
