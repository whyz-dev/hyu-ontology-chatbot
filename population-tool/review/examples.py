"""사람과 함께 확정한 Few-shot 예제를 읽고 관련 예제를 선택한다."""

from __future__ import annotations

import re
from pathlib import Path

from adapters.storage import read_json, sha256_text, stable_json
from config import PROFILE_ROOT
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, SKOS, XSD

EXAMPLE_PATH = PROFILE_ROOT / "fewshot_examples.json"
RESOURCE_ROOT = "urn:hyu-chatbot:resource:"
ANNOTATION_PREDICATES = {RDF.type, RDFS.label, SKOS.prefLabel, SKOS.altLabel}


def _literal_object(value: Literal) -> dict[str, object]:
    datatype = value.datatype
    if datatype == XSD.boolean:
        return {"kind": "boolean", "value": bool(value.toPython())}
    if datatype == XSD.integer:
        return {"kind": "integer", "value": int(value)}
    if datatype in {XSD.decimal, XSD.double, XSD.float}:
        return {"kind": "decimal", "value": float(value)}
    if datatype == XSD.date:
        return {"kind": "date", "value": str(value)}
    if datatype in {XSD.dateTime, XSD.dateTimeStamp}:
        return {"kind": "datetime", "value": str(value)}
    return {"kind": "string", "value": str(value)}


def _transport_example(raw: dict[str, object], turtle: str) -> str:
    """사람이 관리하는 Turtle 정답을 구조화 출력의 wire format으로 투영한다."""
    graph = Graph().parse(data=turtle, format="turtle")
    resources = sorted(
        {
            value
            for triple in graph
            for value in (triple[0], triple[2])
            if isinstance(value, URIRef) and str(value).startswith(RESOURCE_ROOT)
        },
        key=str,
    )
    entities = []
    for resource in resources:
        labels = [
            str(value)
            for predicate in (SKOS.prefLabel, RDFS.label)
            for value in graph.objects(resource, predicate)
        ]
        identifiers = [
            str(value) for value in graph.objects(resource, DCTERMS.identifier)
        ]
        entities.append(
            {
                "entity_id": str(resource).removeprefix(RESOURCE_ROOT),
                "label": labels[0]
                if labels
                else str(resource).removeprefix(RESOURCE_ROOT),
                "identifier": identifiers[0] if identifiers else None,
                "types": sorted(
                    str(value)
                    for value in graph.objects(resource, RDF.type)
                    if isinstance(value, URIRef)
                ),
            }
        )

    facts = []
    locator = str(raw["source_locator"])
    for subject, predicate, obj in graph:
        if not str(subject).startswith(RESOURCE_ROOT):
            continue
        if predicate in ANNOTATION_PREDICATES or predicate == DCTERMS.identifier:
            continue
        if isinstance(obj, URIRef) and str(obj).startswith(RESOURCE_ROOT):
            wire_object: dict[str, object] = {
                "kind": "entity",
                "value": str(obj).removeprefix(RESOURCE_ROOT),
            }
        elif isinstance(obj, Literal):
            wire_object = _literal_object(obj)
        else:
            continue
        facts.append(
            {
                "subject": str(subject).removeprefix(RESOURCE_ROOT),
                "predicate": str(predicate),
                "object": wire_object,
                "evidence_locator": locator,
            }
        )
    facts.sort(
        key=lambda item: (
            str(item["subject"]),
            str(item["predicate"]),
            stable_json(item["object"]),
        )
    )
    return stable_json(
        {
            "entities": entities,
            "facts": facts,
            "no_fact_locators": [] if facts else [locator],
        }
    )


def _load_example(raw: dict[str, object]) -> dict[str, object]:
    # 예제 내용은 검증하지 않지만 metadata가 profile 밖의 파일을 읽게 하지는 않는다.
    relative = Path(str(raw["ontology_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe few-shot ontology path: {relative}")
    turtle = (PROFILE_ROOT / relative).read_text(encoding="utf-8")
    result = dict(raw)
    result["ontology_path"] = relative.as_posix()
    result["ontology_ttl"] = turtle.strip() + "\n"
    result["ontology_sha256"] = sha256_text(result["ontology_ttl"])
    result["transport_json"] = _transport_example(raw, result["ontology_ttl"])
    return result


def load_examples() -> list[dict[str, object]]:
    """확정 예제 파일을 수정하거나 승인 판정하지 않고 그대로 읽는다."""
    payload = read_json(EXAMPLE_PATH)
    return [_load_example(dict(item)) for item in payload["examples"]]


def example_set_hash() -> str:
    """실행에 사용한 확정 예제 집합을 provenance hash로 남긴다."""
    return sha256_text(stable_json(load_examples()))


def _ngrams(value: str, size: int = 3) -> set[str]:
    normalized = re.sub(r"\s+", "", value.lower())
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    }


def _text_similarity(left: str, right: str) -> float:
    left_set, right_set = _ngrams(left), _ngrams(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def select_examples(
    examples: list[dict[str, object]],
    limit: int = 4,
    evidence_text: str = "",
    required_category: str | None = None,
) -> list[dict[str, object]]:
    relevant = [item for item in examples if item.get("category") != "no_fact"]
    ranked = sorted(
        relevant,
        key=lambda item: (
            -_text_similarity(evidence_text, str(item.get("text", ""))),
            str(item["example_id"]),
        ),
    )
    # 안내/제목을 억지 사실로 만드는 경향을 줄이기 위해 no-fact 반례를 항상 넣는다.
    no_fact = next(
        (item for item in examples if item.get("category") == "no_fact"),
        None,
    )
    selected = ranked[: max(0, limit - (1 if no_fact else 0))]
    # 표 일정처럼 구조가 명확한 경우에만 호출자가 category를 강제한다. 날짜가
    # 들어간 줄글 규칙에 schedule 예제를 무조건 넣으면 관계 전체가 시간표 구조로
    # 과적합되므로 evidence 문구만 보고 자동 강제하지 않는다.
    if required_category and not any(
        item.get("category") == required_category for item in selected
    ):
        required = next(
            (item for item in ranked if item.get("category") == required_category),
            None,
        )
        if required is not None:
            if selected:
                selected[-1] = required
            else:
                selected.append(required)
    if no_fact is not None:
        selected.append(no_fact)
    return selected
