"""Pipeline 단계 사이에서 주고받는 핵심 데이터 계약을 정의한다."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class EvidenceUnit(TypedDict):
    """본문 block과 표 전체를 함께 보존한 페이지 단위 근거."""

    schema_version: str
    unit_id: str
    kind: str
    document_id: str
    pdf_page: int | None
    printed_page: int | None
    locator: str
    text: str
    context: dict[str, object]
    token_ids: list[object]
    source_status: str
    source_hash: str
    content_hash: str
    source_issues: NotRequired[list[object]]


class OntologyTerm(TypedDict):
    """Upstream 용어 또는 LLM이 제안한 local schema 용어."""

    term_id: str
    kind: str
    iri: str | None
    label: str
    definition: str
    domains: list[str]
    ranges: list[str]
    upstream_parent_iri: str | None
    implicit: NotRequired[bool]


class EntityCandidate(TypedDict):
    """근거 원문에 명시적으로 등장하는 instance 후보."""

    entity_id: str
    label: str
    identifier: str | None
    types: list[str]


class FactObject(TypedDict):
    """Entity 참조 또는 datatype literal로 사용할 fact의 목적어."""

    kind: str
    value: object


class TemporalScope(TypedDict):
    """근거가 명시한 fact의 적용 시점 또는 기간."""

    label: str
    start: str | None
    end: str | None


class FactCandidate(TypedDict):
    """정확한 인용문으로 grounding된 subject-predicate-object 후보."""

    subject: str
    predicate: str
    object: FactObject
    evidence_locator: str
    evidence_quote: str
    confidence: float
    temporal_scope: TemporalScope | None


class PopulationCandidate(TypedDict, total=False):
    """한 evidence unit에서 추출한 ontology population 결과."""

    schema_version: str
    unit_id: str
    unit_content_hash: str
    terms: list[OntologyTerm]
    entities: list[EntityCandidate]
    facts: list[FactCandidate]
    no_fact_locators: list[str]
    coverage: dict[str, object]
    generation: dict[str, object]
    status: str
    issues: list[str]
    created_at: str
    model_digest: str
    example_set_hash: str
