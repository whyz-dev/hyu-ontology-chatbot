"""한 사용자 질문의 전체 ontology QA 흐름을 조정한다.

질의 생성 → ontology 용어 정합 → 안전한 SPARQL 1회 실행 → 근거 답변 생성을 연결한다.
각 단계의 실패를 별도 상태로 반환하며, 실패 시 원문 검색이나 임의 답변 같은 우회 경로는
사용하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.exceptions import OutputParserException
from models.ontology import OntologyStore
from models.schemas import (
    AnswerGenerationError,
    QueryGenerationError,
    ResolvedTerm,
    TermResolutionError,
)
from models.settings import SETTINGS

from .answer import AnswerGenerator
from .query import QueryGenerator
from .terms import TermResolver

EMPTY_RESULT = {"variables": [], "rows": [], "row_count": 0}
NO_RESULT_ANSWER = (
    "현재 수동 온톨로지에서 질문의 조건을 모두 만족하는 근거를 찾지 못했습니다. "
    "실제 정보가 없다는 뜻은 아니며, 현재 지식 범위에 포함되지 않았을 수 있습니다."
)
QUERY_FAILURE_ANSWER = (
    "질문을 안전한 온톨로지 질의로 변환하지 못했습니다. "
    "지식 부재가 아니라 질의 생성 실패입니다."
)
TERM_FAILURE_ANSWER = (
    "질문의 핵심 표현을 현재 온톨로지 용어에 안전하게 연결하지 못했습니다. "
    "실제 정보가 없다는 뜻은 아니며, 현재 어휘 범위에 포함되지 않았을 수 있습니다."
)
EXECUTION_FAILURE_ANSWER = (
    "생성한 온톨로지 질의를 실행하지 못했습니다. 지식 부재가 아니라 실행 실패입니다."
)
ANSWER_FAILURE_ANSWER = (
    "온톨로지 검색 결과는 얻었지만 검증 가능한 답변 형식으로 만들지 못했습니다."
)


class OntologyQA:
    """View가 호출하는 ontology QA 진입 Controller.

    입력은 독립적인 자연어 질문 하나이고, 출력은 답변과 중간 산출물을 포함한 trace
    사전이다. 생성 단계의 LLM 재시도와 무관하게, 정합이 끝난 SPARQL 자체는 정확히 한 번만
    실행한다.
    """

    def __init__(
        self,
        ontology_path: Path = SETTINGS.ontology_path,
        *,
        model: str = SETTINGS.model,
        ollama_url: str = SETTINGS.ollama_url,
    ) -> None:
        """온톨로지 저장소와 단계별 Controller를 동일 실행 설정으로 구성한다."""

        self.store = OntologyStore(ontology_path)
        self.model_name = model
        self.query_generator = QueryGenerator(
            self.store,
            model=model,
            ollama_url=ollama_url,
        )
        self.term_resolver = TermResolver(
            self.store,
            model=model,
            ollama_url=ollama_url,
        )
        self.answer_generator = AnswerGenerator(model=model, ollama_url=ollama_url)

    def _trace(
        self,
        *,
        status: str,
        question: str,
        answer: str,
        insufficient_knowledge: bool | None,
        draft_query: str | None,
        query_attempts: int,
        query: str | None = None,
        query_executed: bool = False,
        resolved_terms: list[ResolvedTerm] | None = None,
        result: dict[str, object] | None = None,
        used_result_rows: list[int] | None = None,
        answer_attempts: int = 0,
        query_generation_error: str | None = None,
        term_resolution_error: str | None = None,
        query_execution_error: str | None = None,
        answer_generation_error: str | None = None,
    ) -> dict[str, object]:
        """성공과 실패를 View가 같은 방식으로 표시하도록 trace를 정규화한다.

        ``insufficient_knowledge``는 검색 근거의 충분성만 표현한다. 질의 생성·용어 정합·
        실행 오류에는 참/거짓을 추정하지 않고 ``None``을 사용해 지식 부재와 구분한다.
        """

        return {
            "status": status,
            "question": question,
            "answer": answer,
            "insufficient_knowledge": insufficient_knowledge,
            "used_result_rows": used_result_rows or [],
            "draft_query": draft_query,
            "query_attempts": query_attempts,
            "query": query,
            "query_executed": query_executed,
            "answer_attempts": answer_attempts,
            "query_generation_error": query_generation_error,
            "term_resolution_error": term_resolution_error,
            "query_execution_error": query_execution_error,
            "answer_generation_error": answer_generation_error,
            "resolved_terms": [term.as_dict() for term in (resolved_terms or [])],
            "result": result or dict(EMPTY_RESULT),
            "model": self.model_name,
            "ontology": str(self.store.path),
            "ontology_triples": self.store.triple_count,
        }

    def ask(self, question: str) -> dict[str, object]:
        """질문을 처리하고 사용자 답변 및 전체 실행 trace를 반환한다.

        빈 질문은 호출 오류로 거부한다. 이후 단계에서 예상된 실패는 예외를 View까지
        전파하지 않고 단계별 status로 변환한다. 단, ontology 초기화 등 요청 흐름 바깥의
        실행 오류는 View의 최상위 오류 처리에 맡긴다.
        """

        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        # 1) Qwen이 placeholder를 포함한 SPARQL 초안을 만들고 Controller가 안전성을
        # 검증한다. 이 단계에서는 아직 질의를 실행하지 않는다.
        try:
            draft, draft_query, query_attempts = self.query_generator.create(question)
        except QueryGenerationError as error:
            draft_query = error.draft.sparql if error.draft is not None else None
            return self._trace(
                status="query_generation_failed",
                question=question,
                answer=QUERY_FAILURE_ANSWER,
                insufficient_knowledge=None,
                draft_query=draft_query,
                query_attempts=2,
                query_generation_error=str(error),
            )

        # 2) 질문 표현을 기존 ontology IRI에만 정합한 뒤 placeholder를 치환한다.
        # 정합 실패를 정보 부재로 간주하거나 문자열 검색으로 우회하지 않는다.
        try:
            terms = self.term_resolver.resolve(question, draft.terms)
            query = self.store.substitute_terms(draft_query, terms)
        except (TermResolutionError, OutputParserException, ValueError) as error:
            return self._trace(
                status="term_unresolved",
                question=question,
                answer=TERM_FAILURE_ANSWER,
                insufficient_knowledge=None,
                draft_query=draft_query,
                query_attempts=query_attempts,
                term_resolution_error=str(error),
            )

        # 3) 치환된 SELECT를 마지막으로 다시 검사한 후 여기서 정확히 한 번 실행한다.
        try:
            query = self.store.safe_select_query(query)
            result = self.store.execute(query)
        except Exception as error:  # noqa: BLE001 - RDFLib 예외 계층이 불안정하다.
            return self._trace(
                status="query_execution_failed",
                question=question,
                answer=EXECUTION_FAILURE_ANSWER,
                insufficient_knowledge=None,
                draft_query=draft_query,
                query_attempts=query_attempts,
                query=query,
                resolved_terms=terms,
                query_execution_error=f"{type(error).__name__}: {error}",
            )

        # 실행 성공과 0행 결과는 서로 다른 상태다. 0행은 현재 ontology 범위에서 근거가
        # 부족하다는 뜻일 뿐, 실제 세계에서 명제가 거짓임을 의미하지 않는다.
        if not result["rows"]:
            return self._trace(
                status="no_result",
                question=question,
                answer=NO_RESULT_ANSWER,
                insufficient_knowledge=True,
                draft_query=draft_query,
                query_attempts=query_attempts,
                query=query,
                query_executed=True,
                resolved_terms=terms,
                result=result,
            )

        # 4) 검색 행이 있을 때만 Qwen이 자연어 답변과 인용 행을 생성한다.
        try:
            grounded_answer, answer_attempts = self.answer_generator.create(
                question, query, result
            )
        except AnswerGenerationError as error:
            return self._trace(
                status="answer_generation_failed",
                question=question,
                answer=ANSWER_FAILURE_ANSWER,
                insufficient_knowledge=None,
                draft_query=draft_query,
                query_attempts=query_attempts,
                query=query,
                query_executed=True,
                resolved_terms=terms,
                result=result,
                answer_attempts=2,
                answer_generation_error=str(error),
            )

        # LLM도 결과가 질문 조건을 충족하지 않는다고 판단할 수 있다. 이 경우 실행
        # 자체는 성공했어도 no_result로 표현하고 인용 행을 남기지 않는다.
        return self._trace(
            status=(
                "no_result" if grounded_answer.insufficient_knowledge else "answered"
            ),
            question=question,
            answer=self.answer_generator.append_source_pages(grounded_answer, result),
            insufficient_knowledge=grounded_answer.insufficient_knowledge,
            used_result_rows=grounded_answer.used_result_rows,
            draft_query=draft_query,
            query_attempts=query_attempts,
            query=query,
            query_executed=True,
            answer_attempts=answer_attempts,
            resolved_terms=terms,
            result=result,
        )
