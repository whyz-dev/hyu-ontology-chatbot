"""질의 생성, 용어 정합, 답변 생성 단계의 구조화 데이터 계약.

LLM의 JSON Schema 출력과 결정론적 Python 코드 사이의 경계다. 특히 placeholder,
근거 행, answerability 불변식을 모델 단계에서 검사해 잘못된 LLM 출력이 다음 단계로
전파되지 않게 한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryTerm(BaseModel):
    """SPARQL에서 실제 IRI로 확정하기 전인 질문 속 domain resource.

    LLM이 온톨로지 resource IRI를 추측하지 않도록 임시 IRI와 원문 표현, 예상 class만
    보관한다. 실제 IRI 선택은 별도의 용어 정합 단계가 담당한다.
    """

    model_config = ConfigDict(extra="forbid")

    placeholder_iri: str = Field(
        pattern=r"^urn:query-term:[A-Za-z0-9_-]+$",
        description="SPARQL에 <...> 형태로 넣는 고유 placeholder IRI",
    )
    mention: str = Field(min_length=1, description="질문에 실제로 나온 표현")
    expected_type: str | None = Field(
        default=None,
        description="가능하면 hyu:Department 같은 ontology class CURIE",
    )


class QueryDraft(BaseModel):
    """Qwen이 만든 읽기 전용 SPARQL 초안과 정합 대상 용어 목록."""

    model_config = ConfigDict(extra="forbid")

    sparql: str = Field(description="SELECT SPARQL query")
    terms: list[QueryTerm] = Field(default_factory=list)

    @model_validator(mode="after")
    def placeholders_are_unique(self) -> QueryDraft:
        """한 placeholder가 서로 다른 표현을 가리키는 모호한 초안을 거부한다."""

        values = [term.placeholder_iri for term in self.terms]
        if len(values) != len(set(values)):
            raise ValueError("term placeholders must be unique")
        return self


class TermChoice(BaseModel):
    """후보 목록 안에서 LLM이 고른 하나의 ontology resource.

    ``selected_iri=None``은 억지로 가장 가까운 후보를 고르지 않고 정합 실패로
    종료할 수 있게 하는 명시적 선택이다.
    """

    placeholder_iri: str
    selected_iri: str | None = Field(
        description="후보 중 선택한 IRI. 적합한 후보가 없으면 null"
    )
    confidence: float = Field(ge=0, le=1)
    reason: str


class TermRefinement(BaseModel):
    """한 질의에 포함된 모든 placeholder에 대한 LLM 선택 결과."""

    choices: list[TermChoice]


class GroundedAnswer(BaseModel):
    """SPARQL 결과 행만으로 작성한 최종 답변과 근거 위치."""

    answer: str = Field(
        min_length=1,
        max_length=600,
        description="검색 결과만 사용한 3문장 이내의 한국어 답변",
    )
    used_result_rows: list[int] = Field(
        default_factory=list,
        description="답변에 실제 사용한 검색 결과 행의 0-based index",
    )
    insufficient_knowledge: bool = Field(
        description="검색 결과만으로 질문에 답할 수 없으면 true"
    )

    @model_validator(mode="after")
    def evidence_rows_match_answerability(self) -> GroundedAnswer:
        """답변 가능성 판단과 근거 행 인용이 서로 모순되지 않게 한다."""

        if self.insufficient_knowledge and self.used_result_rows:
            raise ValueError("An insufficient answer must not cite result rows")
        if not self.insufficient_knowledge and not self.used_result_rows:
            raise ValueError("A grounded answer must cite at least one result row")
        return self


@dataclass(frozen=True)
class TermCandidate:
    """질문의 한 표현과 연결할 수 있는 기존 ontology resource 후보.

    ``matched_label``은 실제 유사도 계산에 사용된 label이고 ``label``은 화면에 표시할
    대표 이름이다. 둘을 분리해 별칭으로 검색된 경우도 추적할 수 있다.
    """

    iri: str
    label: str
    matched_label: str
    types: tuple[str, ...]
    score: float

    def as_dict(self) -> dict[str, object]:
        """LLM 입력과 trace에 사용할 JSON 호환 표현을 반환한다."""

        return asdict(self)


@dataclass(frozen=True)
class ResolvedTerm:
    """placeholder를 기존 ontology IRI로 확정한 결과.

    ``method``는 결정론적 exact match인지 후보 중 LLM 선택인지 구분해 trace에 남긴다.
    """

    placeholder_iri: str
    mention: str
    selected_iri: str
    selected_label: str
    method: Literal["exact", "llm"]
    score: float

    def as_dict(self) -> dict[str, object]:
        """사용자에게 노출할 trace용 JSON 호환 표현을 반환한다."""

        return asdict(self)


class TermResolutionError(ValueError):
    """질문 표현을 기존 ontology resource에 안전하게 연결하지 못한 경우."""


class QueryGenerationError(ValueError):
    """재시도 후에도 안전한 SELECT 질의를 만들지 못한 경우."""

    def __init__(self, message: str, draft: QueryDraft | None) -> None:
        """오류 문구와 마지막 구조화 초안을 함께 보존한다."""

        super().__init__(message)
        self.draft = draft


class AnswerGenerationError(ValueError):
    """검색 결과가 있지만 검증 가능한 답변 형식을 만들지 못한 경우."""
