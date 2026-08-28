"""후보의 원문 grounding과 draft/published RDF 문법을 검증한다."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import cache
from pathlib import Path

from adapters.rdf import build_draft_graph
from adapters.sources import (
    application_profile_iris,
    application_profile_terms,
    load_application_profile_graph,
)
from adapters.storage import read_jsonl, sha256_file, write_json
from config import CANDIDATE_SCHEMA_VERSION
from domain.models import EvidenceUnit, PopulationCandidate
from domain.vocabulary import (
    CANDIDATE_IRI_ROOT,
    OBJECT_KINDS,
    ONTOLOGY_TERMS,
)
from rdflib import Graph, URIRef
from rdflib.compare import isomorphic
from rdflib.exceptions import ParserError

NUMBER_PATTERN = re.compile(r"\d+")
LOCAL_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
TIME_IN_DATE = {
    "http://www.w3.org/2006/time#inXSDDate",
    "http://www.w3.org/2006/time#inXSDDateTimeStamp",
}
TIME_BOUNDARY = {
    "http://www.w3.org/2006/time#hasBeginning",
    "http://www.w3.org/2006/time#hasEnd",
}
INSTANT = "http://www.w3.org/2006/time#Instant"
INTERVAL_TYPES = {
    "http://www.w3.org/2006/time#Interval",
    "http://www.w3.org/2006/time#ProperInterval",
    "urn:hyu-chatbot:vocab:CourseRegistrationPeriod",
    "urn:hyu-chatbot:vocab:AcademicTerm",
}
TYPE_PARENTS = {
    "http://www.w3.org/2006/time#Instant": {
        "http://www.w3.org/2006/time#TemporalEntity"
    },
    "http://www.w3.org/2006/time#Interval": {
        "http://www.w3.org/2006/time#TemporalEntity"
    },
    "http://www.w3.org/2006/time#ProperInterval": {
        "http://www.w3.org/2006/time#Interval"
    },
}
XSD_KIND = {
    "http://www.w3.org/2001/XMLSchema#boolean": "boolean",
    "http://www.w3.org/2001/XMLSchema#date": "date",
    "http://www.w3.org/2001/XMLSchema#dateTime": "datetime",
    "http://www.w3.org/2001/XMLSchema#dateTimeStamp": "datetime",
    "http://www.w3.org/2001/XMLSchema#decimal": "decimal",
    "http://www.w3.org/2001/XMLSchema#integer": "integer",
    "http://www.w3.org/2001/XMLSchema#string": "string",
}


@cache
def _profile_contract() -> dict[str, dict[str, object]]:
    graph = load_application_profile_graph()
    for child, parents in list(TYPE_PARENTS.items()):
        TYPE_PARENTS[child] = set(parents)
    for child, parent in graph.subject_objects(
        URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    ):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            TYPE_PARENTS.setdefault(str(child), set()).add(str(parent))
    return {str(item["iri"]): dict(item) for item in application_profile_terms()}


def _type_compatible(actual: set[str], expected: set[str]) -> bool:
    frontier = list(actual)
    closure = set(actual)
    while frontier:
        current = frontier.pop()
        for parent in TYPE_PARENTS.get(current, set()):
            if parent not in closure:
                closure.add(parent)
                frontier.append(parent)
    return bool(closure & expected)


def _numbers(value: object) -> list[str]:
    # OCR의 `9.22`는 소수 9.22가 아니라 9월 22일일 수 있다. 숫자를 구성 요소로
    # 비교해 ISO 날짜 `2026-09-22`와 원문의 점 표기를 같은 근거로 취급한다.
    return [str(int(number)) for number in NUMBER_PATTERN.findall(str(value))]


def _compact(value: object) -> str:
    return re.sub(r"[\s|`*_#]+", "", str(value))


def _date_supported(value: object, quote: str, grounding: str) -> bool:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError:
        return False
    direct_numbers = Counter(_numbers(quote))
    required = Counter([str(parsed.month), str(parsed.day)])
    if any(direct_numbers[number] < count for number, count in required.items()):
        return False
    numbers = set(_numbers(grounding))
    if str(parsed.year) in numbers:
        return True
    match = re.search(r"(\d{4})\s*학년도(?:\s*2\s*학기)?", grounding)
    if match is None:
        return False
    academic_year = int(match.group(1))
    expected = academic_year + 1 if 1 <= parsed.month <= 3 else academic_year
    return parsed.year == expected


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.utcoffset() == timedelta(hours=9) else None


DATETIME_MENTION = re.compile(
    r"(?:(\d{4})\s*[.년]\s*)?(\d{1,2})\s*[.월]\s*(\d{1,2})\s*"
    r"(?:[.일])?\s*(?:\([^)]*\))?\s*(\d{1,2}):(\d{2})"
)
SAME_DAY_DATETIME_RANGE = re.compile(
    r"(?:(\d{4})\s*[.년]\s*)?(\d{1,2})\s*[.월]\s*(\d{1,2})\s*"
    r"(?:[.일])?\s*(?:\([^)]*\))?\s*(\d{1,2}):(\d{2})\s*"
    r"[-~]\s*(\d{1,2}):(\d{2})"
)


def _source_datetimes(quote: str, grounding: str) -> set[str]:
    academic = re.search(r"(\d{4})\s*학년도", grounding)
    result: set[str] = set()
    for match in DATETIME_MENTION.finditer(quote):
        month = int(match.group(2))
        year = (
            int(match.group(1))
            if match.group(1)
            else (int(academic.group(1)) + (1 if month <= 3 else 0) if academic else 0)
        )
        if not year:
            continue
        day = date(year, month, int(match.group(3)))
        hour = int(match.group(4))
        minute = int(match.group(5))
        if hour == 24 and minute == 0:
            day += timedelta(days=1)
            hour = 0
        elif not 0 <= hour <= 23:
            continue
        value = datetime(
            day.year,
            day.month,
            day.day,
            hour,
            minute,
            tzinfo=timezone(timedelta(hours=9)),
        )
        result.add(value.isoformat(timespec="seconds"))
    for match in SAME_DAY_DATETIME_RANGE.finditer(quote):
        month = int(match.group(2))
        year = (
            int(match.group(1))
            if match.group(1)
            else (int(academic.group(1)) + (1 if month <= 3 else 0) if academic else 0)
        )
        if not year:
            continue
        for hour_group, minute_group in ((4, 5), (6, 7)):
            day = date(year, month, int(match.group(3)))
            hour = int(match.group(hour_group))
            minute = int(match.group(minute_group))
            if hour == 24 and minute == 0:
                day += timedelta(days=1)
                hour = 0
            elif not 0 <= hour <= 23:
                continue
            value = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                minute,
                tzinfo=timezone(timedelta(hours=9)),
            )
            result.add(value.isoformat(timespec="seconds"))
    return result


def _known_reference(
    reference: object,
    local_terms: dict[str, dict[str, object]],
    allowed_iris: set[str],
) -> bool:
    value = str(reference)
    return value in local_terms or value in allowed_iris


def validate_candidate(unit: EvidenceUnit, candidate: PopulationCandidate) -> list[str]:
    # 이 검증기는 LLM을 호출하지 않는다. 같은 입력은 항상 같은 판정을 내린다.
    issues: list[str] = []
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        issues.append("unsupported candidate schema")
    if candidate.get("unit_id") != unit.get("unit_id"):
        issues.append("candidate unit ID differs from evidence unit")
    if candidate.get("unit_content_hash") != unit.get("content_hash"):
        issues.append("candidate content hash differs from evidence unit")
    # 단일 application profile 밖의 IRI를 모델이 임의로 끌어오는 것을 막는다.
    allowed_iris = set(ONTOLOGY_TERMS)
    allowed_iris.update(application_profile_iris())
    context = unit.get("context", {})
    evidence = context.get("evidence", {}) if isinstance(context, dict) else {}
    raw_locators = context.get("locators", []) if isinstance(context, dict) else []
    allowed_locators = {str(value) for value in raw_locators}

    def locator_value(locator: str, field: str) -> str:
        item = evidence.get(locator) if isinstance(evidence, dict) else None
        return str(item.get(field, "")) if isinstance(item, dict) else ""

    terms = candidate.get("terms", [])
    entities = candidate.get("entities", [])
    facts = candidate.get("facts", [])
    if not all(isinstance(item, dict) for item in [*terms, *entities, *facts]):
        return [*issues, "terms, entities and facts must contain objects"]

    term_map = {str(item.get("term_id")): item for item in terms}
    entity_map = {str(item.get("entity_id")): item for item in entities}
    contract = _profile_contract()
    if terms:
        issues.append("population candidate terms must be empty")
    if len(entity_map) != len(entities):
        issues.append("entity IDs are empty or duplicated")

    for entity_id, entity in entity_map.items():
        label = str(entity.get("label", ""))
        if not LOCAL_ID_PATTERN.fullmatch(entity_id):
            issues.append(f"entity {entity_id!r} has an invalid internal ID")
        if not label:
            issues.append(f"entity {entity_id!r} needs a label")
        types = entity.get("types")
        if not isinstance(types, list) or not types:
            issues.append(f"entity {entity_id!r} types must be an array")
            continue
        for type_reference in types:
            if not _known_reference(type_reference, term_map, allowed_iris):
                issues.append(
                    f"entity {entity_id!r} has unknown type {type_reference!r}"
                )
            elif (
                str(type_reference) in term_map
                and term_map[str(type_reference)].get("kind") != "class"
            ):
                issues.append(
                    f"entity {entity_id!r} uses non-class type {type_reference!r}"
                )

    seen_facts: set[tuple[str, str, str, str, str]] = set()
    entity_locators: dict[str, set[str]] = {
        entity_id: set() for entity_id in entity_map
    }
    # 숫자·날짜·entity·인용문이 실제 evidence에 존재하는지 fact마다 확인한다.
    for index, fact in enumerate(facts, start=1):
        prefix = f"fact {index}"
        subject = str(fact.get("subject", ""))
        predicate = str(fact.get("predicate", ""))
        locator = str(fact.get("evidence_locator", ""))
        grounding = locator_value(locator, "grounding_text")
        expected_quote = locator_value(locator, "quote")
        if subject not in entity_map:
            issues.append(f"{prefix} has unknown subject {subject!r}")
        else:
            entity_locators[subject].add(locator)
        if not _known_reference(predicate, term_map, allowed_iris):
            issues.append(f"{prefix} has unknown predicate {predicate!r}")
        obj = fact.get("object")
        if not isinstance(obj, dict) or obj.get("kind") not in OBJECT_KINDS:
            issues.append(f"{prefix} has invalid object")
            continue
        if obj["kind"] == "entity":
            object_entity = str(obj.get("value"))
            if object_entity not in entity_map:
                issues.append(
                    f"{prefix} has unknown object entity {obj.get('value')!r}"
                )
            if object_entity == subject:
                issues.append(f"{prefix} is a self-referencing entity relation")
            elif object_entity in entity_locators:
                entity_locators[object_entity].add(locator)
        if obj["kind"] == "integer" and (
            not isinstance(obj.get("value"), int) or isinstance(obj.get("value"), bool)
        ):
            issues.append(f"{prefix} integer object must be an integer")
        if obj["kind"] == "decimal" and (
            not isinstance(obj.get("value"), (int, float))
            or isinstance(obj.get("value"), bool)
        ):
            issues.append(f"{prefix} decimal object must be numeric")
        if obj["kind"] == "boolean" and not isinstance(obj.get("value"), bool):
            issues.append(f"{prefix} boolean object must be true or false")
        if obj["kind"] in {"integer", "decimal"} and (
            not _numbers(obj.get("value"))
            or not set(_numbers(obj.get("value"))) <= set(_numbers(expected_quote))
        ):
            issues.append(f"{prefix} numeric object is absent from evidence")
        if obj["kind"] == "string" and (
            not str(obj.get("value", ""))
            or _compact(obj.get("value", "")) not in _compact(grounding)
        ):
            issues.append(f"{prefix} string object is empty or absent from evidence")
        if obj["kind"] == "date" and not _date_supported(
            obj.get("value"), expected_quote, grounding
        ):
            issues.append(f"{prefix} date object is not supported by its locator")
        if obj["kind"] == "datetime":
            if _parse_time(obj.get("value")) is None:
                issues.append(f"{prefix} datetime object is not a Seoul dateTimeStamp")
            elif str(obj.get("value")) not in _source_datetimes(
                expected_quote, grounding
            ):
                issues.append(f"{prefix} datetime object is absent from evidence")
        if obj["kind"] == "boolean":
            positive = bool(re.search(r"가능|허용|포함|인정|승인|운영", grounding))
            negative = bool(
                re.search(r"불가|금지|제외|없|않|못|미운영|불허|처리되지", grounding)
            )
            if obj.get("value") is True and not positive:
                issues.append(f"{prefix} true value has no positive evidence")
            if obj.get("value") is False and not negative:
                issues.append(f"{prefix} false value has no negative evidence")
        quote = str(fact.get("evidence_quote", ""))
        if not quote or quote != expected_quote:
            issues.append(f"{prefix} quote is not exact evidence text")
        if locator not in allowed_locators or not grounding:
            issues.append(f"{prefix} has unknown evidence locator {locator!r}")
        confidence = fact.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            issues.append(f"{prefix} confidence is outside 0..1")
        if fact.get("temporal_scope") is not None:
            issues.append(f"{prefix} temporal scope must be null in page candidates")

        signature = contract.get(predicate)
        if signature:
            property_kind = str(signature.get("kind"))
            domains = {str(value) for value in signature.get("domains", [])}
            ranges = {str(value) for value in signature.get("ranges", [])}
            subject_types = set(entity_map.get(subject, {}).get("types", []))
            if domains and not _type_compatible(subject_types, domains):
                issues.append(f"{prefix} subject type violates predicate domain")
            if property_kind == "object_property":
                if obj.get("kind") != "entity":
                    issues.append(f"{prefix} object property needs an entity object")
                else:
                    object_types = set(
                        entity_map.get(str(obj.get("value")), {}).get("types", [])
                    )
                    if ranges and not _type_compatible(object_types, ranges):
                        issues.append(f"{prefix} object type violates predicate range")
            elif property_kind == "datatype_property":
                if obj.get("kind") == "entity":
                    issues.append(f"{prefix} datatype property cannot target an entity")
                expected_kinds = {
                    XSD_KIND[value] for value in ranges if value in XSD_KIND
                }
                if expected_kinds and obj.get("kind") not in expected_kinds:
                    issues.append(f"{prefix} literal kind violates predicate range")

        # 관계의 양 끝 개체도 같은 행/문단에서 확인되어야 한다. 다른 표 행에 같은
        # 이름이 있다는 이유만으로 현재 관계가 grounded되는 것을 허용하지 않는다.
        for role, entity_id in (
            ("subject", subject),
            (
                "object",
                str(obj.get("value")) if obj.get("kind") == "entity" else "",
            ),
        ):
            if not entity_id or entity_id not in entity_map:
                continue
            entity = entity_map[entity_id]
            label = str(entity.get("label", ""))
            types = set(entity.get("types", []))
            date_label = INSTANT in types and bool(_numbers(label))
            if label and _compact(label) not in _compact(grounding) and not date_label:
                issues.append(
                    f"{prefix} {role} entity {entity_id!r} is absent from its locator"
                )
            identifier = entity.get("identifier")
            if identifier is not None and _compact(identifier) not in _compact(
                grounding
            ):
                issues.append(
                    f"{prefix} {role} identifier for {entity_id!r} is absent from its locator"
                )

        subject_types = set(entity_map.get(subject, {}).get("types", []))
        if predicate in TIME_IN_DATE and INSTANT not in subject_types:
            issues.append(f"{prefix} applies an XSD date to a non-Instant entity")
        if (
            predicate == "http://www.w3.org/2006/time#inXSDDate"
            and obj.get("kind") != "date"
        ):
            issues.append(f"{prefix} inXSDDate requires a date object")
        if (
            predicate == "http://www.w3.org/2006/time#inXSDDateTimeStamp"
            and obj.get("kind") != "datetime"
        ):
            issues.append(f"{prefix} inXSDDateTimeStamp requires a datetime object")
        if predicate in TIME_BOUNDARY:
            object_types = set(
                entity_map.get(str(obj.get("value")), {}).get("types", [])
            )
            if not subject_types & INTERVAL_TYPES:
                issues.append(f"{prefix} has a non-Interval time boundary subject")
            if INSTANT not in object_types:
                issues.append(f"{prefix} has a non-Instant time boundary object")

        fact_key = (
            subject,
            predicate,
            str(obj.get("kind")),
            str(obj.get("value")),
            locator,
        )
        if fact_key in seen_facts:
            issues.append(f"{prefix} duplicates an earlier triple")
        seen_facts.add(fact_key)

    time_literals = {
        (str(fact.get("subject")), str(fact.get("evidence_locator")))
        for fact in facts
        if str(fact.get("predicate")) in TIME_IN_DATE
        and isinstance(fact.get("object"), dict)
        and fact["object"].get("kind") in {"date", "datetime"}
    }
    for index, fact in enumerate(facts, start=1):
        obj = fact.get("object")
        if not isinstance(obj, dict) or obj.get("kind") != "entity":
            continue
        object_id = str(obj.get("value"))
        predicate = str(fact.get("predicate"))
        object_types = set(entity_map.get(object_id, {}).get("types", []))
        if (
            predicate in TIME_BOUNDARY | {"urn:hyu-chatbot:vocab:effectiveAt"}
            and INSTANT in object_types
            and (object_id, str(fact.get("evidence_locator"))) not in time_literals
        ):
            issues.append(
                f"fact {index} points to an Instant without a same-locator time literal"
            )

    # 시간 범위를 두 개의 독립 Instant로만 남기면 질의에서 기간의 시작/종료를
    # 복원할 수 없다. 원문에 시각 범위가 있는 locator는 두 boundary와 literal을
    # 모두 가져야 완전한 population으로 인정한다.
    for locator in allowed_locators:
        quote = locator_value(locator, "quote")
        if not re.search(r"\d{1,2}:\d{2}.*[-~].*\d{1,2}:\d{2}", quote):
            continue
        locator_facts = [
            fact for fact in facts if str(fact.get("evidence_locator")) == locator
        ]
        if not locator_facts:
            continue
        predicates = {str(fact.get("predicate")) for fact in locator_facts}
        if not TIME_BOUNDARY <= predicates:
            issues.append(
                f"{locator}: datetime range needs both hasBeginning and hasEnd"
            )
        literal_count = sum(
            1
            for fact in locator_facts
            if str(fact.get("predicate"))
            == "http://www.w3.org/2006/time#inXSDDateTimeStamp"
        )
        if literal_count < 2:
            issues.append(f"{locator}: datetime range needs two time literals")

    # Entity 이름과 식별자도 실제로 그 entity를 사용하는 locator 안에서 확인한다.
    for entity_id, entity in entity_map.items():
        grounding = "\n".join(
            locator_value(locator, "grounding_text")
            for locator in sorted(entity_locators.get(entity_id, set()))
        )
        label = str(entity.get("label", ""))
        types = set(entity.get("types", []))
        date_label = INSTANT in types and bool(_numbers(label))
        if label and _compact(label) not in _compact(grounding) and not date_label:
            issues.append(f"entity {entity_id!r} label is absent from its locator")
        identifier = entity.get("identifier")
        if identifier is not None and _compact(identifier) not in _compact(grounding):
            issues.append(f"entity {entity_id!r} identifier is absent from its locator")

    fact_locators = {str(fact.get("evidence_locator", "")) for fact in facts}
    raw_no_fact_locators = candidate.get("no_fact_locators", [])
    if not isinstance(raw_no_fact_locators, list) or not all(
        isinstance(value, str) for value in raw_no_fact_locators
    ):
        issues.append("candidate no-fact locators must be an array of strings")
        raw_no_fact_locators = []
    no_fact_locators = {str(value) for value in raw_no_fact_locators}
    if len(no_fact_locators) != len(raw_no_fact_locators):
        issues.append("candidate no-fact locators contain duplicates")
    if no_fact_locators - allowed_locators:
        issues.append("candidate no-fact locators contain unknown locators")
    coverage = candidate.get("coverage")
    if not isinstance(coverage, dict):
        issues.append("candidate coverage is missing")
    else:
        if set(coverage.get("target_locators", [])) != allowed_locators:
            issues.append("coverage target locators differ from evidence")
        if set(coverage.get("fact_locators", [])) != fact_locators:
            issues.append("coverage fact locators differ from facts")
        if set(coverage.get("no_fact_locators", [])) != no_fact_locators:
            issues.append("coverage no-fact locators differ from candidate")
        unresolved = set(coverage.get("unresolved_locators", []))
        failed = set(coverage.get("failed_locators", []))
        if fact_locators & no_fact_locators:
            issues.append("a locator cannot contain facts and be marked no-fact")
        if fact_locators | no_fact_locators != allowed_locators:
            issues.append("not every evidence locator is classified")
        if unresolved or failed or coverage.get("complete") is not True:
            issues.append("candidate coverage is incomplete")

    return issues


def validate_rdf(path: Path) -> tuple[bool, str]:
    # 현재 단계는 SHACL 규칙 검증이 아니라 Turtle 구문과 RDF graph 구성만 검사한다.
    graph = Graph()
    try:
        graph.parse(path, format="turtle")
    except (OSError, ParserError, SyntaxError, ValueError) as error:
        return False, f"RDF parse failed: {error}"
    return True, f"RDF parsed successfully: {len(graph)} triples"


def validate_run(
    units_path: Path,
    candidates_path: Path,
    draft_root: Path,
    report_path: Path,
    *,
    run_id: str,
    reviewed_candidates: dict[str, dict[str, object]],
    mutable_inputs: list[Path],
) -> dict[str, object]:
    units = {str(item["unit_id"]): item for item in read_jsonl(units_path)}
    candidates = reviewed_candidates
    issues = []
    for unit_id, unit in units.items():
        candidate = candidates.get(unit_id)
        if candidate is None:
            issues.append(f"{unit_id}: missing successful candidate")
            continue
        issues.extend(
            f"{unit_id}: {item}" for item in validate_candidate(unit, candidate)
        )
        ttl_path = draft_root / str(unit["document_id"]) / f"{unit_id}.ttl"
        if not candidate.get("facts"):
            if ttl_path.exists():
                issues.append(f"{unit_id}: no-fact page must not have a draft TTL")
            continue
        if not ttl_path.exists():
            issues.append(f"{unit_id}: missing page draft TTL")
            continue
        conforms, report = validate_rdf(ttl_path)
        if not conforms:
            issues.append(f"{unit_id}: {report}")
            continue
        actual = Graph().parse(ttl_path, format="turtle")
        expected = build_draft_graph(
            run_id,
            unit,
            candidate,
            str(candidate.get("model_digest", "")),
        )
        if not isomorphic(actual, expected):
            issues.append(f"{unit_id}: draft TTL differs from the reviewed candidate")
    extra = [
        path.as_posix() for path in draft_root.rglob("*.ttl") if path.stem not in units
    ]
    issues.extend(f"unexpected draft TTL: {item}" for item in extra)
    report = {
        "schema_version": "population-validation-v2",
        "valid": not issues,
        "counts": {
            "units": len(units),
            "successful_candidates": len(candidates),
            "issues": len(issues),
        },
        "issues": issues,
        "inputs": {
            "units_sha256": sha256_file(units_path),
            "candidates_sha256": sha256_file(candidates_path),
            **{
                path.name.replace(".", "_") + "_sha256": (
                    sha256_file(path) if path.exists() else None
                )
                for path in mutable_inputs
            },
        },
    }
    write_json(report_path, report)
    return report


def candidate_iris(graph: Graph) -> list[str]:
    values = set()
    for subject, predicate, obj in graph:
        for value in (subject, predicate, obj):
            if isinstance(value, URIRef) and str(value).startswith(CANDIDATE_IRI_ROOT):
                values.add(str(value))
    return sorted(values)
