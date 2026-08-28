"""Evidence unit에서 ontology population 후보를 구조화 출력으로 추출한다."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

from adapters.ollama import OllamaClient
from adapters.sources import application_profile_terms, ontology_term
from adapters.storage import sha256_text, stable_json
from config import CANDIDATE_SCHEMA_VERSION
from domain.models import EvidenceUnit, PopulationCandidate
from domain.vocabulary import INTERNAL_PROFILE_IRIS, OBJECT_KINDS, ONTOLOGY_TERMS
from review.examples import select_examples

APPLICATION_PROFILE_RULES = """- 교과목은 schema:Course, 학기·분반 강좌는 schema:CourseInstance로 구분한다.
- 대학·학과·전공은 ORG, 학생·교과목 분류는 HYU의 SKOS 기반 분류 class를 사용한다.
- 하루 일정은 time:Instant, 기간은 time:Interval과 시작·종료 Instant로 표현한다.
- 허용·금지·학점·횟수 제한은 hyu:AcademicRule 하위 규칙 instance로 표현한다.
- 학번·학년·캠퍼스·학기·조건·예외·부정을 근거보다 넓히거나 생략하지 않는다.
- 아래 contract에 없는 class/property가 필요하면 새 용어를 만들지 말고 해당 사실을 생략한다."""


def _available_terms(graph: object) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for iri in ONTOLOGY_TERMS:
        if iri.startswith(("http://www.w3.org/ns/prov#", "http://purl.org/dc/terms/")):
            continue
        records.setdefault(iri, ontology_term(graph, iri))
    for term in application_profile_terms():
        iri = str(term["iri"])
        if iri in INTERNAL_PROFILE_IRIS:
            continue
        # Schema.org처럼 upstream이 owl:Object/DatatypeProperty를 직접 선언하지
        # 않는 용어는 application profile의 명시적 계약을 우선한다.
        records[iri] = dict(term)
    return list(records.values())


def _reference_schema(description: str) -> dict[str, object]:
    # Ollama가 pattern/anyOf를 토큰마다 grammar로 검사하면 Qwen 3.5 생성이
    # 수 분간 정체된다. JSON의 모양만 여기서 제한하고 실제 IRI/내부 ID
    # 허용 여부는 validate_candidate가 결정적으로 검사한다.
    return {
        "type": "string",
        "description": description,
    }


def extraction_schema(
    target_count: int,
    graph: object | None = None,
    examples: list[dict[str, object]] | None = None,
    target_locators: list[str] | None = None,
) -> dict[str, Any]:
    # 시간 범위 하나도 Interval·시작/종료 Instant·규칙·학생/행위까지 최대
    # 6개 entity와 7개 fact가 필요하다. 이보다 작은 상한은 올바른 답 자체를
    # JSON Schema 단계에서 막으므로 locator당 충분한 여유를 둔다.
    # 한 행에 서로 다른 두 신청 구간이 병기된 표도 있어 단일 locator의 상한은
    # 더 넉넉해야 한다. 여러 locator 배치는 locator당 8/12 비율로 제한한다.
    max_entities = max(16, min(80, target_count * 8))
    max_facts = max(24, min(120, target_count * 12))
    selected_terms = (
        _selected_contract_terms(graph, examples)
        if graph is not None and examples is not None
        else []
    )
    class_iris = [
        str(item["iri"]) for item in selected_terms if item["kind"] == "class"
    ]
    type_schema = _reference_schema("canonical class의 전체 IRI")
    if class_iris:
        type_schema["enum"] = class_iris

    entity = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "maxLength": 64,
                "description": "응답 내부에서만 쓰는 ID. res:, hyu:, URL, URN을 쓰지 않는다.",
            },
            "label": {
                "type": "string",
                "maxLength": 240,
                "description": "근거에 그대로 존재하는 개체 이름. slug나 생성 IRI가 아니다.",
            },
            "identifier": {"type": ["string", "null"], "maxLength": 128},
            "types": {
                "type": "array",
                "items": type_schema,
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["entity_id", "label", "identifier", "types"],
        "additionalProperties": False,
    }
    xsd_kinds = {
        "http://www.w3.org/2001/XMLSchema#boolean": ("boolean", "boolean"),
        "http://www.w3.org/2001/XMLSchema#date": ("date", "string"),
        "http://www.w3.org/2001/XMLSchema#dateTime": ("datetime", "string"),
        "http://www.w3.org/2001/XMLSchema#dateTimeStamp": (
            "datetime",
            "string",
        ),
        "http://www.w3.org/2001/XMLSchema#decimal": ("decimal", "number"),
        "http://www.w3.org/2001/XMLSchema#integer": ("integer", "integer"),
        "http://www.w3.org/2001/XMLSchema#string": ("string", "string"),
    }
    property_groups: dict[tuple[str, str], list[str]] = {}
    for term in selected_terms:
        kind = str(term["kind"])
        if kind == "class":
            continue
        if kind == "object_property":
            object_kind, value_type = "entity", "string"
        else:
            range_iris = [str(value) for value in term.get("ranges", [])]
            object_kind, value_type = next(
                (xsd_kinds[value] for value in range_iris if value in xsd_kinds),
                ("string", "string"),
            )
        property_groups.setdefault((object_kind, value_type), []).append(
            str(term["iri"])
        )

    locator_schema: dict[str, object] = {
        "type": "string",
        "maxLength": 100,
        "description": "근거를 포함한 BLOCK 또는 ROW locator",
    }
    if target_locators:
        locator_schema["enum"] = target_locators

    def fact_branch(
        predicates: list[str], object_kind: str, value_type: str
    ) -> dict[str, object]:
        predicate_schema: dict[str, object] = {"type": "string"}
        if predicates:
            predicate_schema["enum"] = predicates
        return {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "entities 배열의 entity_id",
                },
                "predicate": predicate_schema,
                "object": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": [object_kind]},
                        "value": {
                            "type": value_type,
                            "description": (
                                "kind=entity이면 entities의 entity_id, "
                                "아니면 근거의 typed literal"
                            ),
                        },
                    },
                    "required": ["kind", "value"],
                    "additionalProperties": False,
                },
                "evidence_locator": locator_schema,
            },
            "required": ["subject", "predicate", "object", "evidence_locator"],
            "additionalProperties": False,
        }

    branches = [
        fact_branch(predicates, object_kind, value_type)
        for (object_kind, value_type), predicates in property_groups.items()
    ]
    fact: dict[str, object] = (
        {"oneOf": branches} if branches else fact_branch([], "string", "string")
    )
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": entity,
                "maxItems": max_entities,
            },
            "facts": {"type": "array", "items": fact, "maxItems": max_facts},
            "no_fact_locators": {
                "type": "array",
                "items": locator_schema,
                "maxItems": target_count,
            },
        },
        "required": ["entities", "facts", "no_fact_locators"],
        "additionalProperties": False,
    }


def _selected_contract_terms(
    graph: object, examples: list[dict[str, object]]
) -> list[dict[str, object]]:
    """이번 few-shot이 실제 사용하는 contract 용어와 그 signature만 고른다."""
    terms = _available_terms(graph)
    by_iri = {str(item["iri"]): item for item in terms}
    example_turtle = "\n".join(str(item.get("ontology_ttl", "")) for item in examples)
    selected: set[str] = set()
    annotation_iris = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
        "http://www.w3.org/2004/02/skos/core#altLabel",
    }
    for iri in by_iri:
        compact = next(
            (
                prefix + iri[len(root) :]
                for prefix, root in PREFIXES.items()
                if iri.startswith(root)
            ),
            None,
        )
        if iri not in annotation_iris and (
            f"<{iri}>" in example_turtle
            or (compact is not None and compact in example_turtle)
        ):
            selected.add(iri)

    # property의 domain/range가 목록 밖이면 올바른 entity type을 고를 수 없다.
    for iri in list(selected):
        term = by_iri[iri]
        selected.update(
            value
            for value in [*term.get("domains", []), *term.get("ranges", [])]
            if str(value) in by_iri
        )
    return [by_iri[iri] for iri in by_iri if iri in selected]


def _ontology_context(graph: object, examples: list[dict[str, object]]) -> str:
    return "\n".join(
        _term_signature(item) for item in _selected_contract_terms(graph, examples)
    )


def _term_signature(term: dict[str, object]) -> str:
    domains = ", ".join(str(value) for value in term.get("domains", [])) or "-"
    ranges = ", ".join(str(value) for value in term.get("ranges", [])) or "-"
    return (
        f"- {term['iri']} | {term['kind']} | {term.get('label', '')} "
        f"| domain={domains} | range={ranges}"
    )


def extraction_prompt(
    unit: EvidenceUnit,
    graph: object,
    examples: list[dict[str, object]],
    target_locators: list[str],
) -> str:
    fewshot_sections = []
    for index, item in enumerate(examples, start=1):
        section = (
            f"[예시 {index}: {item['example_id']}]\n"
            f"근거:\n{item['text']}\n"
            f"정답 RDF/OWL (Turtle):\n{item['ontology_ttl']}"
        )
        if index == 1:
            section += f"같은 Turtle의 JSON 전송 형식:\n{item['transport_json']}\n"
        fewshot_sections.append(section)
    fewshot = "\n\n".join(fewshot_sections)
    targets = "\n".join(f"- {value}" for value in target_locators)
    grounding = _locator_grounding(unit)
    target_evidence = "\n\n".join(
        f"[TARGET {locator}]\n{grounding.get(locator, '')}"
        for locator in target_locators
    )
    return f"""지정된 한양대학교 학사안내 원문만 canonical ontology에 population하라.
section과 표 header는 문맥일 뿐이고, 사실은 TARGET locator에서만 가져온다.

[모델링 기준]
{APPLICATION_PROFILE_RULES}

[출력 규칙]
1. few-shot Turtle의 resource와 type/label은 JSON `entities`로, 나머지 domain triple은 `facts`로 옮긴다. rdf:type, rdfs:label, skos:prefLabel, xsd datatype은 fact predicate가 아니다.
2. 모든 fact의 subject와 entity object ID를 `entities`에 정확히 한 번 선언하고 contract의 domain/range에 맞는 type을 쓴다. object property의 object는 반드시 `kind=entity`이고 `value`는 entity_id다. `appliesTo*`와 `targets*`의 subject는 AcademicRule이다.
3. label은 TARGET에 연속해서 존재하는 원문 구절 그대로 쓰며 `기간`, `시작`, `종료`, `일시`를 덧붙이지 않는다. fact에는 해당 TARGET locator를 쓴다.
4. 날짜·시간 범위는 Interval→hasBeginning/hasEnd→Instant로 표현한다. 날짜는 inXSDDate/date, 시간이 있으면 서울 현지시각의 inXSDDateTimeStamp/datetime이다. 모든 Instant에는 같은 locator의 literal이 필요하며 24:00은 다음 날 00:00+09:00이다.
5. 2학기 일정의 7~12월은 학년도와 같은 해, 이어지는 1~3월은 다음 해다.
6. 각 TARGET은 fact를 하나 이상 갖거나 `no_fact_locators`에 정확히 한 번 들어간다. contract로 표현할 수 없거나 모호한 대상과 보통의 table_header는 no_fact다.
7. few-shot의 구조만 참고하고 예제 ID나 사실을 복사하지 않는다.

[이번 호출에 필요한 canonical ontology contract]
{_ontology_context(graph, examples)}

[확정 few-shot]
{fewshot}

[이번 호출의 추출 대상 locator]
{targets}

[추출 대상 원문]
{target_evidence}
"""


PREFIXES = {
    "dcterms:": "http://purl.org/dc/terms/",
    "hyu:": "urn:hyu-chatbot:vocab:",
    "org:": "http://www.w3.org/ns/org#",
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "schema:": "https://schema.org/",
    "skos:": "http://www.w3.org/2004/02/skos/core#",
    "time:": "http://www.w3.org/2006/time#",
    "xsd:": "http://www.w3.org/2001/XMLSchema#",
}
ANNOTATION_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
}


def _expand_reference(value: object) -> str:
    text = str(value).strip()
    for prefix, root in PREFIXES.items():
        if text.startswith(prefix):
            return root + text[len(prefix) :]
    return text


def _internal_id(value: object) -> str:
    text = str(value).strip()
    for prefix in ("res:", "urn:hyu-chatbot:resource:"):
        text = text.removeprefix(prefix)
    if text.startswith(("http://", "https://", "urn:")):
        text = re.split(r"[/#:]", text.rstrip("/"))[-1]
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:64]
    if not text or not text[0].isalpha():
        text = f"entity-{text or 'unknown'}"
    return text


def _scalar(value: object) -> object:
    # 일부 모델은 JSON scalar를 {"entity_id": ...}처럼 한 번 더 감싼다.
    # 알려진 wrapper만 벗기며 새로운 값을 추론하지 않는다.
    while isinstance(value, dict):
        for key in ("entity_id", "value", "text", "ko"):
            if key in value:
                value = value[key]
                break
        else:
            return ""
    return value


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _compact_text(value: object) -> str:
    return re.sub(r"[\s|`*_#]+", "", str(value))


def _number_tokens(value: object) -> set[str]:
    return {str(int(item)) for item in re.findall(r"\d+", str(value))}


def _locator_evidence(unit: EvidenceUnit) -> dict[str, dict[str, object]]:
    context = unit.get("context", {})
    raw = context.get("evidence", {}) if isinstance(context, dict) else {}
    return {
        str(locator): dict(item)
        for locator, item in raw.items()
        if isinstance(item, dict)
    }


def _locator_quotes(unit: EvidenceUnit) -> dict[str, str]:
    return {
        locator: str(item.get("quote", "")).strip()
        for locator, item in _locator_evidence(unit).items()
        if str(item.get("quote", "")).strip()
    }


def _locator_grounding(unit: EvidenceUnit) -> dict[str, str]:
    return {
        locator: str(item.get("grounding_text", item.get("quote", ""))).strip()
        for locator, item in _locator_evidence(unit).items()
    }


def _calendar_date(value: str, source: str) -> str:
    """2학기 학사일정의 월을 표 제목 기준 calendar year로 정규화한다."""
    parsed = date.fromisoformat(value)
    match = re.search(r"(\d{4})\s*학년도\s*2\s*학기\s*학사일정", source)
    if not match:
        return value
    academic_year = int(match.group(1))
    if 7 <= parsed.month <= 12:
        expected_year = academic_year
    elif 1 <= parsed.month <= 3:
        expected_year = academic_year + 1
    else:
        return value
    return parsed.replace(year=expected_year).isoformat()


def _date_evidence_label(value: str, quote: str) -> str:
    parsed = date.fromisoformat(value)
    pattern = re.compile(rf"0?{parsed.month}\s*월\s*0?{parsed.day}\s*일")
    match = pattern.search(quote)
    return match.group(0) if match else value


def _date_supported(value: str, quote: str, grounding: str) -> bool:
    """월·일은 직접 인용에서, 연도만 표/section 문맥에서 복원한다."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    direct_numbers = Counter(str(int(item)) for item in re.findall(r"\d+", str(quote)))
    required = Counter([str(parsed.month), str(parsed.day)])
    if any(direct_numbers[value] < count for value, count in required.items()):
        return False
    numbers = _number_tokens(grounding)
    if str(parsed.year) in numbers:
        return True
    match = re.search(r"(\d{4})\s*학년도(?:\s*2\s*학기)?", grounding)
    if match is None:
        return False
    academic_year = int(match.group(1))
    expected = academic_year + 1 if 1 <= parsed.month <= 3 else academic_year
    return parsed.year == expected


DATETIME_MENTION = re.compile(
    r"(?:(\d{4})\s*[.년]\s*)?(\d{1,2})\s*[.월]\s*(\d{1,2})\s*"
    r"(?:[.일])?\s*(?:\([^)]*\))?\s*(\d{1,2}):(\d{2})"
)
SAME_DAY_DATETIME_RANGE = re.compile(
    r"(?:(\d{4})\s*[.년]\s*)?(\d{1,2})\s*[.월]\s*(\d{1,2})\s*"
    r"(?:[.일])?\s*(?:\([^)]*\))?\s*(\d{1,2}):(\d{2})\s*"
    r"[-~]\s*(\d{1,2}):(\d{2})"
)


def _normalize_datetime(value: str) -> str:
    """문서의 서울 현지시각과 24:00 표기를 유효한 dateTimeStamp로 바꾼다."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?(Z|[+-]\d{2}:\d{2})?",
        value,
    )
    if match is None:
        raise ValueError("not an ISO local datetime")
    day = date.fromisoformat(match.group(1))
    hour = int(match.group(2))
    minute = int(match.group(3))
    second = int(match.group(4) or 0)
    if hour == 24 and minute == 0 and second == 0:
        day += timedelta(days=1)
        hour = 0
    elif not 0 <= hour <= 23:
        raise ValueError("invalid hour")
    offset = match.group(5)
    zone = timezone(timedelta(hours=9)) if offset is None else None
    parsed = datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        second,
        tzinfo=zone,
    )
    if offset is not None:
        parsed = datetime.fromisoformat(
            f"{day.isoformat()}T{hour:02d}:{minute:02d}:{second:02d}{offset}"
        )
    if parsed.utcoffset() != timedelta(hours=9):
        raise ValueError("datetime is outside the Seoul document timezone")
    return parsed.isoformat(timespec="seconds")


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
        local = (
            f"{year:04d}-{month:02d}-{int(match.group(3)):02d}T"
            f"{int(match.group(4)):02d}:{int(match.group(5)):02d}"
        )
        try:
            result.add(_normalize_datetime(local))
        except ValueError:
            continue
    # 종료일이 생략된 같은 날 범위는 앞 날짜를 그대로 사용한다.
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
            local = (
                f"{year:04d}-{month:02d}-{int(match.group(3)):02d}T"
                f"{int(match.group(hour_group)):02d}:"
                f"{int(match.group(minute_group)):02d}"
            )
            try:
                result.add(_normalize_datetime(local))
            except ValueError:
                continue
    return result


def normalize_result(
    unit: EvidenceUnit,
    raw: dict[str, object],
    graph: object,
) -> dict[str, list[dict[str, object]]]:
    """LLM의 prefix·wrapper 차이를 하나의 candidate 포맷으로 정규화한다."""
    available = _available_terms(graph)
    kinds = {str(item["iri"]): str(item["kind"]) for item in available}
    signatures = {str(item["iri"]): item for item in available}
    parents = {
        str(item["iri"]): {str(value) for value in item.get("parents", [])}
        for item in available
    }

    def compatible(actual: set[str], expected: set[str]) -> bool:
        closure = set(actual)
        pending = list(actual)
        while pending:
            current = pending.pop()
            for parent in parents.get(current, set()):
                if parent not in closure:
                    closure.add(parent)
                    pending.append(parent)
        return not expected or bool(closure & expected)

    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for iri in kinds:
        local_name = re.split(r"[/#:]", iri.rstrip("/"))[-1]
        if local_name in aliases and aliases[local_name] != iri:
            ambiguous.add(local_name)
        else:
            aliases[local_name] = iri
    for local_name in ambiguous:
        aliases.pop(local_name, None)

    def canonical_reference(value: object) -> str:
        expanded = _expand_reference(value)
        return aliases.get(expanded, expanded)

    class_iris = {iri for iri, kind in kinds.items() if kind == "class"}
    property_iris = {iri for iri, kind in kinds.items() if kind != "class"}
    source = str(unit.get("text", ""))
    allowed_locators = {
        str(value) for value in unit.get("context", {}).get("locators", [])
    }
    locator_quotes = _locator_quotes(unit)
    locator_grounding = _locator_grounding(unit)
    grounded_source = "\n".join(locator_grounding.values())
    normalized_source = _normalized_text(grounded_source)
    compact_source = _compact_text(grounded_source)

    entities: dict[str, dict[str, object]] = {}
    raw_entities = raw.get("entities", [])
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            entity_id = _internal_id(item.get("entity_id", ""))
            types = [
                expanded
                for value in item.get("types", [])
                if (expanded := canonical_reference(value)) in class_iris
            ]
            if not types:
                continue
            label = str(item.get("label", "")).strip()
            is_instant = "http://www.w3.org/2006/time#Instant" in types
            date_label = bool(
                is_instant
                and _number_tokens(label)
                and _number_tokens(label) <= _number_tokens(source)
            )
            if not label or (
                _compact_text(label) not in compact_source and not date_label
            ):
                continue
            identifier = item.get("identifier")
            if (
                identifier is not None
                and _normalized_text(identifier) not in normalized_source
            ):
                identifier = None
            entities[entity_id] = {
                "entity_id": entity_id,
                "label": label,
                "identifier": identifier,
                "types": list(dict.fromkeys(types)),
            }

    raw_facts = raw.get("facts", [])
    if not isinstance(raw_facts, list):
        raw_facts = []

    # 모델이 boundary Instant를 fact에서 참조하면서 entities에서 빠뜨리는 경우가 있다.
    # 근거에 존재하는 유효한 date fact에 한해서만 Instant entity를 결정적으로 복원한다.
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        predicate = canonical_reference(item.get("predicate", ""))
        obj = item.get("object")
        locator = str(item.get("evidence_locator", "")).strip()
        if (
            predicate != "http://www.w3.org/2006/time#inXSDDate"
            or not isinstance(obj, dict)
            or obj.get("kind") != "date"
            or locator not in locator_quotes
        ):
            continue
        value = _scalar(obj.get("value"))
        if not isinstance(value, str):
            continue
        try:
            value = _calendar_date(value, source)
            date.fromisoformat(value)
        except ValueError:
            continue
        grounding = locator_grounding.get(locator, "")
        if not _date_supported(value, locator_quotes[locator], grounding):
            continue
        subject = _internal_id(item.get("subject", ""))
        if subject not in entities:
            entities[subject] = {
                "entity_id": subject,
                "label": _date_evidence_label(value, locator_quotes[locator]),
                "identifier": None,
                "types": ["http://www.w3.org/2006/time#Instant"],
            }

    facts: list[dict[str, object]] = []
    if raw_facts:
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            subject = _internal_id(item.get("subject", ""))
            predicate = canonical_reference(item.get("predicate", ""))
            locator = str(item.get("evidence_locator", "")).strip()
            quote = locator_quotes.get(locator)
            grounding = locator_grounding.get(locator, "")
            if locator not in allowed_locators or not quote or not grounding:
                continue
            if subject not in entities or predicate not in property_iris:
                continue
            if predicate in ANNOTATION_PREDICATES:
                continue
            obj = item.get("object")
            if not isinstance(obj, dict) or obj.get("kind") not in OBJECT_KINDS:
                continue
            kind = str(obj["kind"])
            value = _scalar(obj.get("value"))
            # Qwen이 ISO literal을 올바르게 만들고도 generic string으로 태깅하는
            # 경우만 predicate에 따라 전송 형식을 고친다. 값 자체는 바꾸지 않는다.
            if (
                kind == "string"
                and predicate == "http://www.w3.org/2006/time#inXSDDate"
            ):
                kind = "date"
            elif (
                kind == "string"
                and predicate == "http://www.w3.org/2006/time#inXSDDateTimeStamp"
            ):
                kind = "datetime"
            if kind == "entity":
                value = _internal_id(value)
                if value not in entities or value == subject:
                    continue
            elif kind == "date":
                if not isinstance(value, str):
                    continue
                try:
                    value = _calendar_date(value, source)
                    date.fromisoformat(value)
                except ValueError:
                    continue
                if not _date_supported(value, quote, grounding):
                    continue
            elif kind == "datetime":
                if not isinstance(value, str):
                    continue
                try:
                    value = _normalize_datetime(value)
                except ValueError:
                    continue
                if value not in _source_datetimes(quote, grounding):
                    continue
            elif (
                kind in {"integer", "decimal"}
                and not isinstance(value, (int, float))
                or kind == "boolean"
                and not isinstance(value, bool)
                or kind == "string"
                and not isinstance(value, str)
            ):
                continue
            if kind == "string" and (
                not value or _compact_text(value) not in _compact_text(grounding)
            ):
                continue
            if kind in {"integer", "decimal"} and not _number_tokens(
                value
            ) <= _number_tokens(quote):
                continue
            subject_types = set(entities[subject]["types"])
            signature = signatures[predicate]
            domains = {str(value) for value in signature.get("domains", [])}
            ranges = {str(value) for value in signature.get("ranges", [])}
            if not compatible(subject_types, domains):
                continue
            if signature["kind"] == "object_property":
                if kind != "entity" or not compatible(
                    set(entities[str(value)]["types"]), ranges
                ):
                    continue
            elif kind == "entity":
                continue
            if (
                predicate
                in {
                    "http://www.w3.org/2006/time#inXSDDate",
                    "http://www.w3.org/2006/time#inXSDDateTimeStamp",
                }
                and "http://www.w3.org/2006/time#Instant" not in subject_types
            ):
                continue
            if predicate in {
                "http://www.w3.org/2006/time#hasBeginning",
                "http://www.w3.org/2006/time#hasEnd",
            }:
                interval_types = {
                    "http://www.w3.org/2006/time#Interval",
                    "http://www.w3.org/2006/time#ProperInterval",
                    "urn:hyu-chatbot:vocab:CourseRegistrationPeriod",
                    "urn:hyu-chatbot:vocab:AcademicTerm",
                }
                object_types = (
                    set(entities[str(value)]["types"]) if kind == "entity" else set()
                )
                if not subject_types & interval_types or (
                    "http://www.w3.org/2006/time#Instant" not in object_types
                ):
                    continue
            if predicate == "urn:hyu-chatbot:vocab:effectiveDuring":
                rule_types = {
                    "urn:hyu-chatbot:vocab:AcademicRule",
                    "urn:hyu-chatbot:vocab:PermissionRule",
                    "urn:hyu-chatbot:vocab:ProhibitionRule",
                    "urn:hyu-chatbot:vocab:LimitRule",
                    "urn:hyu-chatbot:vocab:CreditRecognitionRule",
                }
                temporal_types = {
                    "http://www.w3.org/2006/time#TemporalEntity",
                    "http://www.w3.org/2006/time#Instant",
                    "http://www.w3.org/2006/time#Interval",
                    "http://www.w3.org/2006/time#ProperInterval",
                    "urn:hyu-chatbot:vocab:CourseRegistrationPeriod",
                    "urn:hyu-chatbot:vocab:AcademicTerm",
                }
                object_types = (
                    set(entities[str(value)]["types"]) if kind == "entity" else set()
                )
                if not subject_types & rule_types or not object_types & temporal_types:
                    continue
            facts.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": {"kind": kind, "value": value},
                    "evidence_locator": locator,
                    "evidence_quote": quote,
                    "confidence": 1.0,
                    "temporal_scope": None,
                }
            )
    # 모델이 같은 triple을 여러 번 반복해도 첫 번째 grounded assertion만 남긴다.
    unique_facts: list[dict[str, object]] = []
    seen_facts: set[tuple[str, str, str, str, str]] = set()
    for fact in facts:
        obj = fact["object"]
        key = (
            str(fact["subject"]),
            str(fact["predicate"]),
            str(obj["kind"]),
            str(obj["value"]),
            str(fact["evidence_locator"]),
        )
        if key in seen_facts:
            continue
        seen_facts.add(key)
        unique_facts.append(fact)
    facts = unique_facts
    # rdf:type/label만 덩그러니 남은 모델 산출물을 사실로 오인하지 않는다. 어떤
    # assertion에서도 참조하지 않은 entity도 draft에 남기지 않는다.
    referenced_entities = {str(fact["subject"]) for fact in facts}
    referenced_entities.update(
        str(fact["object"]["value"])
        for fact in facts
        if fact["object"]["kind"] == "entity"
    )
    entities = {
        entity_id: entity
        for entity_id, entity in entities.items()
        if entity_id in referenced_entities
    }
    return {"terms": [], "entities": list(entities.values()), "facts": facts}


def _remove_reversed_intervals(
    entities: dict[str, dict[str, object]],
    facts: list[dict[str, object]],
) -> list[dict[str, object]]:
    """OCR 열 순서가 뒤집힌 행에서 종료일보다 늦은 시작일을 기간으로 만들지 않는다."""
    dates = {
        str(fact["subject"]): date.fromisoformat(str(fact["object"]["value"]))
        for fact in facts
        if fact["predicate"] == "http://www.w3.org/2006/time#inXSDDate"
        and fact["object"]["kind"] == "date"
    }
    beginnings = {
        str(fact["subject"]): str(fact["object"]["value"])
        for fact in facts
        if fact["predicate"] == "http://www.w3.org/2006/time#hasBeginning"
        and fact["object"]["kind"] == "entity"
    }
    endings = {
        str(fact["subject"]): str(fact["object"]["value"])
        for fact in facts
        if fact["predicate"] == "http://www.w3.org/2006/time#hasEnd"
        and fact["object"]["kind"] == "entity"
    }
    invalid_intervals = {
        interval
        for interval, begin in beginnings.items()
        if interval in endings
        and begin in dates
        and endings[interval] in dates
        and dates[begin] > dates[endings[interval]]
    }
    if not invalid_intervals:
        return facts
    invalid_boundaries = {
        value
        for interval in invalid_intervals
        for value in (beginnings[interval], endings[interval])
    }
    for entity_id in invalid_intervals | invalid_boundaries:
        entities.pop(entity_id, None)
    return [
        fact
        for fact in facts
        if str(fact["subject"]) not in invalid_intervals | invalid_boundaries
        and not (
            fact["object"]["kind"] == "entity"
            and str(fact["object"]["value"]) in invalid_intervals | invalid_boundaries
        )
    ]


def _chunk_locators(
    locators: list[str], locator_quotes: dict[str, str], max_chars: int = 1200
) -> list[list[str]]:
    """연속 locator를 문맥을 잃지 않는 작은 출력 묶음으로 나눈다."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for locator in locators:
        length = len(locator_quotes.get(locator, ""))
        if current and current_chars + length > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(locator)
        current_chars += length
    if current:
        batches.append(current)
    return batches


def extract_unit(
    client: OllamaClient,
    model: str,
    unit: EvidenceUnit,
    graph: object,
    examples: list[dict[str, object]],
) -> PopulationCandidate:
    # 페이지 문맥은 유지하고 출력 대상만 작은 묶음으로 제한한다.
    locators = [str(value) for value in unit.get("context", {}).get("locators", [])]
    locator_quotes = _locator_quotes(unit)
    locator_grounding = _locator_grounding(unit)
    blocks = [value for value in locators if "-block-" in value]
    rows = [value for value in locators if "-row-" in value]
    batches = _chunk_locators(blocks, locator_quotes)
    grouped_rows: list[str] = []
    for locator in rows:
        evidence = _locator_evidence(unit).get(locator, {})
        quote = locator_quotes.get(locator, "")
        needs_own_call = evidence.get("kind") == "table_header" or bool(
            DATETIME_MENTION.search(quote)
        )
        if needs_own_call:
            if grouped_rows:
                batches.append(grouped_rows)
                grouped_rows = []
            batches.append([locator])
            continue
        grouped_rows.append(locator)
        if len(grouped_rows) == 4:
            batches.append(grouped_rows)
            grouped_rows = []
    if grouped_rows:
        batches.append(grouped_rows)
    if not batches and locators:
        batches = [locators]

    prompts = []
    metadata = []
    selected_example_ids: list[str] = []
    entities: dict[str, dict[str, object]] = {}
    facts: list[dict[str, object]] = []
    no_fact_locators: set[str] = set()
    failed_locators: set[str] = set()
    seen_facts: set[tuple[str, str, str, str, str]] = set()
    for batch in batches:
        target_text = "\n".join(locator_grounding.get(value, "") for value in batch)
        selected = select_examples(
            examples,
            3,
            evidence_text=target_text,
        )
        for item in selected:
            example_id = str(item["example_id"])
            if example_id not in selected_example_ids:
                selected_example_ids.append(example_id)
        prompt = extraction_prompt(unit, graph, selected, batch)
        prompts.append(prompt)
        try:
            result, batch_metadata = client.generate_json(
                model,
                prompt,
                extraction_schema(len(batch), graph, selected, batch),
            )
        except RuntimeError as error:
            failed_locators.update(batch)
            metadata.append(
                {
                    "model": model,
                    "status": "failed",
                    "target_locators": batch,
                    "error": str(error),
                }
            )
            continue
        batch_metadata["target_locators"] = batch
        batch_metadata["raw_entities"] = len(result.get("entities", []))
        batch_metadata["raw_facts"] = len(result.get("facts", []))
        normalized = normalize_result(unit, result, graph)
        # 모델은 페이지 전체를 보지만 이번 호출에서 지정한 locator만 산출해야 한다.
        # 다른 행의 사실을 함께 반환하면 뒤 배치에서 중복되거나 일부 행이 건너뛰어질 수
        # 있으므로, 배치 경계를 코드에서도 강제한다.
        target_set = set(batch)
        normalized["facts"] = [
            fact
            for fact in normalized["facts"]
            if fact.get("evidence_locator") in target_set
        ]
        fact_locators = {str(fact["evidence_locator"]) for fact in normalized["facts"]}
        raw_facts = result.get("facts", [])
        raw_fact_locators = {
            str(item.get("evidence_locator", ""))
            for item in raw_facts
            if isinstance(item, dict)
        }
        raw_no_fact = result.get("no_fact_locators", [])
        raw_no_fact_values = (
            [str(value) for value in raw_no_fact if isinstance(value, str)]
            if isinstance(raw_no_fact, list)
            else []
        )
        raw_no_fact_set = set(raw_no_fact_values)
        classification_issues = []
        if len(raw_no_fact_values) != len(raw_no_fact_set):
            classification_issues.append("duplicate no_fact locator")
        if raw_no_fact_set - target_set:
            classification_issues.append("out-of-batch no_fact locator")
        if raw_no_fact_set & raw_fact_locators:
            classification_issues.append("fact/no_fact overlap")
        batch_no_fact = (raw_no_fact_set & target_set) - fact_locators
        if classification_issues:
            # 모호한 분류를 조용히 정규화하면 불완전한 페이지가 성공 처리된다.
            failed_locators.update(target_set)
            batch_metadata["classification_issues"] = classification_issues
        no_fact_locators.update(batch_no_fact)
        unresolved = target_set - fact_locators - batch_no_fact
        batch_metadata["accepted_entities"] = len(normalized["entities"])
        batch_metadata["accepted_facts"] = len(normalized["facts"])
        batch_metadata["no_fact_locators"] = sorted(batch_no_fact)
        batch_metadata["unresolved_locators"] = sorted(unresolved)
        metadata.append(batch_metadata)
        for entity in normalized["entities"]:
            entity_id = str(entity["entity_id"])
            if entity_id not in entities:
                entities[entity_id] = entity
            else:
                entities[entity_id]["types"] = list(
                    dict.fromkeys(
                        [*entities[entity_id]["types"], *entity.get("types", [])]
                    )
                )
        for fact in normalized["facts"]:
            obj = fact["object"]
            key = (
                str(fact["subject"]),
                str(fact["predicate"]),
                str(obj["kind"]),
                str(obj["value"]),
                str(fact["evidence_locator"]),
            )
            if key in seen_facts:
                continue
            seen_facts.add(key)
            facts.append(fact)

    facts = _remove_reversed_intervals(entities, facts)
    fact_locators = {str(fact["evidence_locator"]) for fact in facts}
    no_fact_locators -= fact_locators
    unresolved_locators = set(locators) - fact_locators - no_fact_locators
    unresolved_locators.update(failed_locators)

    referenced = {str(fact["subject"]) for fact in facts}
    referenced.update(
        str(fact["object"]["value"])
        for fact in facts
        if fact["object"]["kind"] == "entity"
    )
    normalized = {
        "terms": [],
        "entities": [
            entity for entity_id, entity in entities.items() if entity_id in referenced
        ],
        "facts": facts,
        "no_fact_locators": sorted(no_fact_locators),
        "coverage": {
            "target_locators": locators,
            "fact_locators": sorted(fact_locators),
            "no_fact_locators": sorted(no_fact_locators),
            "failed_locators": sorted(failed_locators),
            "unresolved_locators": sorted(unresolved_locators),
            "complete": not unresolved_locators,
        },
    }
    extraction_metadata = {
        "model": model,
        "attempt": 1,
        "batches": metadata,
        "batch_count": len(metadata),
        "failed_batch_count": sum(
            1 for item in metadata if item.get("status") == "failed"
        ),
        "total_duration": sum(
            int(item.get("total_duration") or 0) for item in metadata
        ),
        "prompt_eval_count": sum(
            int(item.get("prompt_eval_count") or 0) for item in metadata
        ),
        "eval_count": sum(int(item.get("eval_count") or 0) for item in metadata),
    }
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "unit_id": unit["unit_id"],
        "unit_content_hash": unit["content_hash"],
        **normalized,
        "generation": {
            "extraction": extraction_metadata,
            # prompt가 바뀐 실행을 같은 결과로 오인하지 않도록 provenance에 남긴다.
            "prompt_hash": sha256_text(stable_json(prompts)),
            "example_ids": selected_example_ids,
        },
    }
