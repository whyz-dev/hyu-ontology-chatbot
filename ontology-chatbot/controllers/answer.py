"""SPARQL 결과를 사용자 답변으로 바꾸고 인용 범위를 검증한다.

질문, 실제 실행한 질의, RDFLib 결과 행을 Qwen에 전달하되 외부 지식이나 추가 검색은
허용하지 않는다. LLM 출력은 ``GroundedAnswer``로 제한하고, 코드가 인용 행 번호와
답변 가능성 표시를 다시 검사한다.
"""

from __future__ import annotations

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from models.schemas import AnswerGenerationError, GroundedAnswer
from models.settings import SETTINGS

from .prompts import ANSWER_PROMPT

INSUFFICIENT_ANSWER_PATTERNS = (
    "확인할 수 없",
    "판단할 수 없",
    "알 수 없",
    "포함되어 있지 않",
    "추가 확인",
)


class AnswerGenerator:
    """검색 결과에만 근거한 답변을 생성하는 마지막 LLM Controller.

    LLM은 자연어 답변과 사용한 결과 행 번호를 함께 반환한다. Controller는 인용 행이
    실제 결과 범위 안에 있는지 검증하고, 근거 부족 문구와 구조화 플래그의 모순을
    거부한다.
    """

    def __init__(self, *, model: str, ollama_url: str) -> None:
        """지정한 Ollama 모델을 구조화 답변 schema에 결합한다."""

        llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=0,
            seed=SETTINGS.seed,
            reasoning=False,
            validate_model_on_init=True,
            num_predict=SETTINGS.answer_max_tokens,
        )
        self.chain = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_PROMPT.system_text),
                ("human", ANSWER_PROMPT.human_text),
            ]
        ) | llm.with_structured_output(GroundedAnswer, method="json_schema")

    def create(
        self, question: str, query: str, result: dict[str, object]
    ) -> tuple[GroundedAnswer, int]:
        """질문과 SPARQL 결과로 검증된 답변을 생성한다.

        반환값은 ``(구조화 답변, LLM 호출 횟수)``다. 형식 또는 인용 검증에 실패하면
        동일한 검색 결과와 오류 피드백으로 한 번만 재시도하며, 두 번째 실패는
        ``AnswerGenerationError``로 상위 조정기에 전달한다.
        """

        feedback = "첫 시도입니다. 이전 오류가 없습니다."
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                # 모델에는 검색된 행 전체를 JSON으로 제공하고, 실제 사용한 행은 출력
                # schema의 used_result_rows로 명시하게 한다.
                answer = self.chain.invoke(
                    {
                        "question": question,
                        "query": query,
                        "results": json.dumps(result, ensure_ascii=False, indent=2),
                        "feedback": feedback,
                    }
                )
                if not isinstance(answer, GroundedAnswer):
                    raise ValueError(  # noqa: TRY004 - 재시도할 출력 검증 오류다.
                        "Answer model did not return a GroundedAnswer"
                    )
                # 존재하지 않는 행을 인용하면 답변 내용과 무관하게 grounding 실패다.
                invalid_rows = [
                    index
                    for index in answer.used_result_rows
                    if index < 0 or index >= int(result["row_count"])
                ]
                if invalid_rows:
                    raise ValueError(
                        f"Answer cited rows outside the result set: {invalid_rows}"
                    )
                # 자연어와 구조화 플래그가 모순되면 UI가 성공으로 오인하지 않도록 거부한다.
                if not answer.insufficient_knowledge and any(
                    pattern in answer.answer for pattern in INSUFFICIENT_ANSWER_PATTERNS
                ):
                    raise ValueError(
                        "Answer text says evidence is insufficient but the flag is false"
                    )
                return answer, attempt
            except (OutputParserException, ValueError) as error:
                last_error = error
                # 새 근거를 찾는 fallback 없이 같은 결과의 표현·인용 오류만 교정한다.
                feedback = (
                    f"이전 출력 오류: {error}. 결과 행으로 답할 수 없으면 "
                    "insufficient_knowledge=true와 used_result_rows=[]를 반환하세요."
                )
        raise AnswerGenerationError(str(last_error or "Unknown answer error"))

    @staticmethod
    def append_source_pages(answer: GroundedAnswer, result: dict[str, object]) -> str:
        """인용된 결과 행의 페이지를 코드로 추출해 답변 뒤에 덧붙인다.

        페이지 표기를 LLM에 맡기지 않으므로 존재하지 않는 출처가 생성되지 않는다.
        근거 부족 답변이나 페이지 binding이 없는 결과는 원문 답변을 그대로 반환한다.
        """

        if answer.insufficient_knowledge:
            return answer.answer
        rows = result.get("rows", [])
        if not isinstance(rows, list):
            return answer.answer
        pages: list[str] = []
        for index in answer.used_result_rows:
            if index >= len(rows) or not isinstance(rows[index], dict):
                continue
            source_page = rows[index].get("sourcePage")
            if not isinstance(source_page, dict):
                continue
            value = source_page.get("value")
            if isinstance(value, str):
                page = value.rsplit(":", 1)[-1]
                if page not in pages:
                    pages.append(page)
        if not pages:
            return answer.answer
        return f"{answer.answer.rstrip()} (근거: {', '.join(pages)})"
