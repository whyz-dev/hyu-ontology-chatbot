"""HYU IRI 정책과 단일 application profile의 허용 vocabulary를 정의한다."""

from __future__ import annotations

# vocab은 재사용 가능한 schema, resource는 실제 개체, candidate는 draft 전용이다.
ONTOLOGY_IRI = "urn:hyu-chatbot:ontology:course-registration:2026-2"
APPLICATION_PROFILE_IRI = "urn:hyu-chatbot:profile:course-registration:v1"
APPLICATION_PROFILE_VERSION = "1.1.0"
VOCAB_ROOT = "urn:hyu-chatbot:vocab:"
RESOURCE_IRI_ROOT = "urn:hyu-chatbot:resource:"
CANDIDATE_IRI_ROOT = "urn:hyu-chatbot:candidate:"
UNIT_IRI_ROOT = "urn:hyu-chatbot:evidence-unit:"
ASSERTION_IRI_ROOT = "urn:hyu-chatbot:assertion:"
ACTIVITY_IRI_ROOT = "urn:hyu-chatbot:activity:"
DOCUMENT_IRI_ROOT = "urn:hyu-chatbot:document:"

# 여러 ontology 중 하나를 고르는 모듈 구조가 아니다. HYU 수강신청 ontology 하나가
# 아래 외부 용어를 고정 vocabulary로 재사용한다. LLM에는 upstream 전체가 아니라 이
# 작은 contract와 application profile의 HYU 용어만 제공한다.
ONTOLOGY_TERMS: tuple[str, ...] = (
    "http://www.w3.org/ns/org#Organization",
    "http://www.w3.org/ns/org#OrganizationalUnit",
    "http://www.w3.org/ns/org#Site",
    "http://www.w3.org/ns/org#subOrganizationOf",
    "http://www.w3.org/ns/org#unitOf",
    "http://www.w3.org/ns/org#hasUnit",
    "http://www.w3.org/ns/org#hasSite",
    "http://www.w3.org/ns/org#classification",
    "http://www.w3.org/2006/time#TemporalEntity",
    "http://www.w3.org/2006/time#Instant",
    "http://www.w3.org/2006/time#Interval",
    "http://www.w3.org/2006/time#ProperInterval",
    "http://www.w3.org/2006/time#hasBeginning",
    "http://www.w3.org/2006/time#hasEnd",
    "http://www.w3.org/2006/time#inXSDDate",
    "http://www.w3.org/2006/time#inXSDDateTimeStamp",
    "http://www.w3.org/ns/prov#Entity",
    "http://www.w3.org/ns/prov#Activity",
    "http://www.w3.org/ns/prov#wasDerivedFrom",
    "http://www.w3.org/ns/prov#wasGeneratedBy",
    "http://www.w3.org/ns/prov#generatedAtTime",
    "http://www.w3.org/ns/prov#wasAssociatedWith",
    "http://purl.org/dc/terms/identifier",
    "http://purl.org/dc/terms/title",
    "http://purl.org/dc/terms/source",
    "http://purl.org/dc/terms/issued",
    "http://purl.org/dc/terms/valid",
    "http://purl.org/dc/terms/isPartOf",
    "http://www.w3.org/2004/02/skos/core#Concept",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#broader",
    "http://www.w3.org/2004/02/skos/core#closeMatch",
    "http://www.w3.org/2004/02/skos/core#exactMatch",
    "https://schema.org/Course",
    "https://schema.org/CourseInstance",
    "https://schema.org/courseCode",
    "https://schema.org/hasCourseInstance",
    "https://schema.org/numberOfCredits",
)

# provenance와 profile 자체를 기술하는 용어는 RDF adapter만 사용하며, domain fact를
# 생성하는 LLM vocabulary에는 노출하지 않는다.
INTERNAL_PROFILE_IRIS = frozenset(
    {
        f"{VOCAB_ROOT}EvidenceUnit",
        f"{VOCAB_ROOT}Assertion",
        *(
            f"{VOCAB_ROOT}{name}"
            for name in (
                "contentHash",
                "sourceHash",
                "sourceStatus",
                "sourceIssue",
                "locator",
                "evidenceLocator",
                "evidenceQuote",
                "confidence",
                "pdfPage",
                "printedPage",
                "modelDigest",
                "modelTag",
                "promptHash",
                "exampleSetHash",
                "reviewDecisionHash",
            )
        ),
    }
)

TERM_KINDS = {"class", "object_property", "datatype_property"}
OBJECT_KINDS = {
    "entity",
    "string",
    "integer",
    "decimal",
    "boolean",
    "date",
    "datetime",
}
