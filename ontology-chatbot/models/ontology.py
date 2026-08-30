"""수동 온톨로지 graph의 로딩, 색인, 질의 실행을 제공한다.

LLM에는 전체 Turtle 대신 compact schema와 기존 resource 후보만 노출한다. 생성된
SPARQL은 이 graph에 실제로 존재하는 vocabulary인지 검사한 뒤, 별도 ``sparql``
모듈의 읽기 전용 실행 경로로 전달한다.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from rapidfuzz import fuzz
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from . import sparql as sparql_ops
from .schemas import TermCandidate

HYU = Namespace("urn:hyu-chatbot:vocab:")
RESOURCE = Namespace("urn:hyu-chatbot:resource:")
SCHEMA = Namespace("https://schema.org/")
TIME = Namespace("http://www.w3.org/2006/time#")
ORG = Namespace("http://www.w3.org/ns/org#")

# resource 검색에서 공식명, 별칭, 일반 label을 같은 후보의 이름 집합으로 취급한다.
LABEL_PREDICATES = (SKOS.prefLabel, SKOS.altLabel, RDFS.label, SCHEMA.name)
LOCAL_VOCAB_PREFIX = str(HYU)
QUERY_TERM_PREFIX = "urn:query-term:"
XSD_LOCAL_NAMES = {
    "boolean",
    "date",
    "dateTime",
    "dateTimeStamp",
    "decimal",
    "duration",
    "gYear",
    "integer",
    "nonNegativeInteger",
    "positiveInteger",
    "string",
    "time",
}
QUERY_NAMESPACES = {
    "rdf": RDF,
    "rdfs": RDFS,
    "owl": OWL,
    "skos": SKOS,
    "xsd": XSD,
    "schema": SCHEMA,
    "time": TIME,
    "org": ORG,
    "dcterms": DCTERMS,
    "hyu": HYU,
}

STANDARD_SCHEMA_LINES = (
    "rdf:type [property] 리소스의 RDF class",
    "rdfs:label [property] 사람이 읽는 이름",
    "skos:prefLabel [property] 공식 선호 이름",
    "skos:altLabel [property] 별칭",
    "schema:Course [class] 교과목",
    "schema:CourseInstance [class] 학기·분반별 개설 강좌",
    "schema:name [property] 이름",
    "schema:courseCode [property] 학수번호",
    "schema:numberOfCredits [property] 교과목 학점",
    "schema:identifier [property] 식별자",
    "schema:hasCourseInstance [property] 교과목의 개설 강좌",
    "schema:provider [property] 제공 조직",
    "schema:description [property] 설명",
    "schema:audience [property] 대상",
    "schema:isPartOf [property] 상위 자원",
    "schema:alternateName [property] 다른 이름",
    "schema:location [property] 장소",
    "schema:startDate [property] 시작 날짜",
    "schema:endDate [property] 종료 날짜",
    "org:unitOf [property] 상위 조직",
    "dcterms:identifier [property] 문서 식별자",
    "dcterms:source [property] 출처",
    "dcterms:isPartOf [property] 상위 문서",
    "time:Instant [class] 시점",
    "time:Interval [class] 시간 구간",
    "time:hasBeginning [property] 구간의 시작 Instant",
    "time:hasEnd [property] 구간의 종료 Instant",
    "time:inXSDDate [property] xsd:date 값",
    "time:inXSDDateTimeStamp [property] 시간대가 있는 xsd:dateTime 값",
)


def _normalize(value: str) -> str:
    """한글·영문·숫자만 남겨 띄어쓰기와 표기 기호 차이를 검색에서 제거한다."""

    value = unicodedata.normalize("NFKC", value).lower().strip()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


class OntologyStore:
    """RDFLib graph와 재사용 가능한 schema·label 색인을 함께 보관한다.

    인스턴스 생성 시 Turtle을 한 번 파싱하고 색인을 고정한다. 매 질문마다 파일을 다시
    읽지 않으며, 모든 후보와 schema 정보는 같은 graph snapshot에서 나온다.
    """

    def __init__(self, ontology_path: Path) -> None:
        """수동 Turtle을 읽고 vocabulary와 resource 검색 색인을 구성한다."""

        if not ontology_path.is_file():
            raise FileNotFoundError(f"Manual ontology is missing: {ontology_path}")
        self.path = ontology_path
        self.graph = Graph().parse(ontology_path, format="turtle")
        self._class_ancestor_cache: dict[URIRef, frozenset[URIRef]] = {}
        self._known_iris = {
            node for node in self.graph.all_nodes() if isinstance(node, URIRef)
        } | {
            predicate
            for predicate in self.graph.predicates()
            if isinstance(predicate, URIRef)
        }
        self._labels = self._build_label_index()
        self._schema_catalog = self._build_schema_catalog()

    @property
    def triple_count(self) -> int:
        """현재 로드된 graph snapshot의 triple 수를 trace용으로 반환한다."""

        return len(self.graph)

    def _build_label_index(self) -> dict[URIRef, set[str]]:
        """여러 label predicate를 resource별 이름 집합으로 역색인한다."""

        labels: dict[URIRef, set[str]] = {}
        for predicate in LABEL_PREDICATES:
            for subject, value in self.graph.subject_objects(predicate):
                if isinstance(subject, URIRef) and isinstance(value, Literal):
                    text = str(value).strip()
                    if text:
                        labels.setdefault(subject, set()).add(text)
        return labels

    def _label(self, iri: URIRef) -> str | None:
        """resource의 여러 이름 중 화면과 후보 목록에 쓸 대표 label을 고른다."""

        values = self._labels.get(iri, set())
        if not values:
            return None
        # 한국어 label, 짧은 label 순으로 사람이 읽기 좋은 이름을 선택한다.
        return min(
            values,
            key=lambda value: (
                not bool(re.search(r"[가-힣]", value)),
                len(value),
                value,
            ),
        )

    @staticmethod
    def _curie(iri: URIRef | str) -> str:
        """알려진 namespace의 긴 IRI를 LLM용 CURIE로 축약한다."""

        value = str(iri)
        prefixes = {
            str(HYU): "hyu:",
            str(RESOURCE): "res:",
            str(SCHEMA): "schema:",
            str(TIME): "time:",
            str(RDF): "rdf:",
            str(RDFS): "rdfs:",
            str(OWL): "owl:",
            str(SKOS): "skos:",
            str(ORG): "org:",
            str(DCTERMS): "dcterms:",
            str(XSD): "xsd:",
        }
        for prefix, short in prefixes.items():
            if value.startswith(prefix):
                return short + value.removeprefix(prefix)
        return f"<{value}>"

    @staticmethod
    def _expand_curie(value: str | None) -> URIRef | None:
        """예상 type 표기를 비교 가능한 URIRef로 확장하고 미지원 표기는 거부한다."""

        if not value:
            return None
        prefixes = {
            "hyu:": str(HYU),
            "res:": str(RESOURCE),
            "schema:": str(SCHEMA),
            "time:": str(TIME),
            "org:": str(ORG),
            "dcterms:": str(DCTERMS),
        }
        for short, prefix in prefixes.items():
            if value.startswith(short):
                return URIRef(prefix + value.removeprefix(short))
        if value.startswith("<") and value.endswith(">"):
            return URIRef(value[1:-1])
        if "://" in value or value.startswith("urn:"):
            return URIRef(value)
        return None

    def _build_schema_catalog(self) -> str:
        """질의 생성에 필요한 class/property 서명만 compact text로 만든다."""

        kinds = (
            (OWL.Class, "class"),
            (OWL.ObjectProperty, "object property"),
            (OWL.DatatypeProperty, "datatype property"),
        )
        lines = list(STANDARD_SCHEMA_LINES)
        for rdf_type, kind in kinds:
            terms = sorted(
                {
                    subject
                    for subject in self.graph.subjects(RDF.type, rdf_type)
                    if isinstance(subject, URIRef)
                    and str(subject).startswith(LOCAL_VOCAB_PREFIX)
                },
                key=str,
            )
            for term in terms:
                label = self._label(term) or term.split(":")[-1]
                domains = ",".join(
                    self._curie(value)
                    for value in self.graph.objects(term, RDFS.domain)
                    if isinstance(value, URIRef)
                )
                ranges = ",".join(
                    self._curie(value)
                    for value in self.graph.objects(term, RDFS.range)
                    if isinstance(value, URIRef)
                )
                signature = ""
                if domains or ranges:
                    signature = f" domain={domains or '?'} range={ranges or '?'}"
                lines.append(f"{self._curie(term)} [{kind}] {label}{signature}")

        # 표준 vocabulary는 이 파일 안에서 실제 predicate로 쓰인 항목도 노출한다.
        # 선언 triple이 없는 schema.org·ORG·DCTERMS property를 Qwen이 놓치지 않게 한다.
        documented = {line.split(" ", 1)[0] for line in lines}
        supported_prefixes = (SCHEMA, TIME, ORG, DCTERMS, RDF, RDFS, SKOS)
        standard_predicates = sorted(
            {
                predicate
                for predicate in self.graph.predicates()
                if isinstance(predicate, URIRef)
                and any(str(predicate).startswith(str(ns)) for ns in supported_prefixes)
            },
            key=str,
        )
        for predicate in standard_predicates:
            curie = self._curie(predicate)
            if curie not in documented:
                lines.append(
                    f"{curie} [property] published graph에서 사용하는 표준 관계"
                )
        return "\n".join(lines)

    def schema_catalog(self) -> str:
        """LLM이 query predicate를 고를 때 사용하는 compact schema 목록."""

        return self._schema_catalog

    def _resource_types(self, iri: URIRef) -> tuple[str, ...]:
        """후보 설명에 포함할 resource의 직접 RDF type을 정렬해 반환한다."""

        return tuple(
            sorted(
                self._curie(value)
                for value in self.graph.objects(iri, RDF.type)
                if isinstance(value, URIRef)
            )
        )

    def _class_ancestors(self, rdf_class: URIRef) -> frozenset[URIRef]:
        """명시된 ``rdfs:subClassOf`` 상위 class를 순환 안전하게 계산한다."""

        cached = self._class_ancestor_cache.get(rdf_class)
        if cached is not None:
            return cached
        ancestors: set[URIRef] = {rdf_class}
        pending = [rdf_class]
        while pending:
            current = pending.pop()
            for parent in self.graph.objects(current, RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        result = frozenset(ancestors)
        self._class_ancestor_cache[rdf_class] = result
        return result

    def _matches_expected_type(
        self, rdf_types: set[URIRef], expected_type: URIRef | None
    ) -> bool:
        """직접 type뿐 아니라 명시적 상속 경로까지 예상 class와 비교한다."""

        if expected_type is None:
            return True
        return any(expected_type in self._class_ancestors(value) for value in rdf_types)

    def is_known_vocabulary_iri(self, iri: str) -> bool:
        """IRI가 graph의 기존 vocabulary이거나 승인된 namespace 자체인지 확인한다.

        ``res:`` 인스턴스 IRI는 LLM이 직접 추측할 수 없으며 반드시 placeholder 정합을
        거쳐야 하므로, graph에 비슷한 값이 있더라도 이 검사에서는 허용하지 않는다.
        """

        namespaces = (RDF, RDFS, OWL, SKOS, XSD, SCHEMA, TIME, ORG, DCTERMS, HYU)
        if iri in {str(namespace) for namespace in namespaces}:
            return True
        value = URIRef(iri)
        if iri.startswith(str(RESOURCE)):
            return False
        return value in self._known_iris

    def validate_query_vocabulary(self, sparql: str) -> None:
        """hallucinated absolute IRI와 local hyu term을 query 실행 전에 거부한다."""

        sparql_ops.validate_query_vocabulary(
            sparql,
            is_known_vocabulary_iri=self.is_known_vocabulary_iri,
            namespaces=QUERY_NAMESPACES,
            query_term_prefix=QUERY_TERM_PREFIX,
            xsd_local_names=XSD_LOCAL_NAMES,
        )

    def term_candidates(
        self,
        mention: str,
        expected_type: str | None = None,
        *,
        limit: int = 5,
    ) -> list[TermCandidate]:
        """label 유사도와 예상 type으로 기존 resource 후보만 반환한다.

        class/property 선언은 schema catalog에서 직접 참조하므로 후보에서 제외한다.
        resource는 명시적 class 상속까지 확인한 뒤 WRatio 점수 45 이상만 안정적으로
        정렬해 반환하며, 이 함수 자체는 신규 IRI를 만들지 않는다.
        """

        normalized_mention = _normalize(mention)
        normalized_search = normalized_mention
        expected_iri = self._expand_curie(expected_type)

        scored: list[TermCandidate] = []
        for iri, labels in self._labels.items():
            # schema class/property는 query draft가 catalog에서 직접 사용한다.
            rdf_types = set(self.graph.objects(iri, RDF.type))
            if rdf_types & {OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty}:
                continue
            if not self._matches_expected_type(rdf_types, expected_iri):
                continue
            label_scores = [
                (fuzz.WRatio(normalized_search, _normalize(label)), label)
                for label in labels
            ]
            score, matched_label = max(label_scores, default=(0.0, ""))
            if score < 45:
                continue
            scored.append(
                TermCandidate(
                    iri=str(iri),
                    label=self._label(iri) or matched_label,
                    matched_label=matched_label,
                    types=self._resource_types(iri),
                    score=round(float(score), 2),
                )
            )

        scored.sort(key=lambda item: (-item.score, item.label, item.iri))
        return scored[:limit]

    @staticmethod
    def substitute_terms(
        sparql: str,
        terms: Sequence[sparql_ops.ResolvedTermLike],
    ) -> str:
        """정합된 기존 IRI로 query placeholder를 모두 치환한다."""

        return sparql_ops.substitute_terms(
            sparql,
            terms,
            query_term_prefix=QUERY_TERM_PREFIX,
        )

    @staticmethod
    def safe_select_query(sparql: str) -> str:
        """원격 접근과 mutation을 막고 결과 행 수를 제한한다."""

        return sparql_ops.safe_select_query(sparql)

    def execute(self, sparql: str) -> dict[str, object]:
        """안전 검사를 포함한 SELECT 실행 결과를 JSON 호환 구조로 반환한다."""

        return sparql_ops.execute_select(
            self.graph,
            sparql,
            term_value=self._term_value,
        )

    def _term_value(self, value: object) -> dict[str, object]:
        """RDF term의 종류·datatype·언어·표시 label을 보존해 직렬화한다."""

        if isinstance(value, URIRef):
            return {
                "kind": "iri",
                "value": str(value),
                "label": self._label(value),
            }
        if isinstance(value, Literal):
            return {
                "kind": "literal",
                "value": str(value),
                "datatype": str(value.datatype) if value.datatype else None,
                "language": value.language,
            }
        if isinstance(value, BNode):
            return {"kind": "blank_node", "value": str(value)}
        return {"kind": "unknown", "value": str(value)}

    def evidence_texts(self) -> list[str]:
        """graph에 명시된 원문 근거 문자열을 반환한다."""

        return [str(value) for value in self.graph.objects(None, HYU.sourceText)]
