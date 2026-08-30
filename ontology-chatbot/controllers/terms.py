"""질문의 domain 표현을 기존 ontology resource IRI에 정합한다.

질의 생성기가 남긴 placeholder마다 Model의 label 색인에서 후보를 찾는다. 유일한 완전
일치는 코드로 확정하고, 모호한 경우에만 Qwen이 허용된 후보 중 하나를 고르게 한다.
새 IRI 생성이나 약한 유사도를 이용한 강제 정합은 허용하지 않는다.
"""

from __future__ import annotations

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from models.ontology import OntologyStore
from models.schemas import (
    QueryTerm,
    ResolvedTerm,
    TermRefinement,
    TermResolutionError,
)
from models.settings import SETTINGS

from .prompts import TERMS_PROMPT

MIN_LLM_TERM_SCORE = 60.0
MIN_LLM_TERM_CONFIDENCE = 0.70


class TermResolver:
    """QueryTerm 목록을 실제 ontology resource 목록으로 바꾸는 Controller.

    입력 순서를 유지한 ``ResolvedTerm`` 목록을 반환하며, 하나라도 안전하게 정합할 수
    없으면 부분 결과를 사용하지 않고 ``TermResolutionError``로 전체 단계를 실패시킨다.
    """

    def __init__(self, store: OntologyStore, *, model: str, ollama_url: str) -> None:
        """후보 검색 저장소와 모호성 해소용 구조화 LLM chain을 구성한다."""

        self.store = store
        llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=0,
            seed=SETTINGS.seed,
            reasoning=False,
            validate_model_on_init=True,
            num_predict=SETTINGS.query_max_tokens,
        )
        self.chain = ChatPromptTemplate.from_messages(
            [
                ("system", TERMS_PROMPT.system_text),
                ("human", TERMS_PROMPT.human_text),
            ]
        ) | llm.with_structured_output(TermRefinement, method="json_schema")

    def resolve(self, question: str, terms: list[QueryTerm]) -> list[ResolvedTerm]:
        """질의 placeholder를 검증된 ontology IRI에 일대일로 정합한다.

        후보가 없거나, LLM이 후보 밖 IRI를 선택하거나, 점수·confidence가 임계값보다
        낮으면 실패한다. 빈 ``terms`` 입력은 LLM을 호출하지 않고 빈 목록을 반환한다.
        """

        resolved: dict[str, ResolvedTerm] = {}
        pending: list[dict[str, object]] = []
        candidate_index: dict[str, list[dict[str, object]]] = {}

        for term in terms:
            # 후보 검색은 질문 mention과 예상 class를 함께 사용하며, 후보가 없으면 임의의
            # resource를 생성하지 않는다.
            candidates = self.store.term_candidates(term.mention, term.expected_type)
            if not candidates:
                raise TermResolutionError(
                    f"No ontology term candidates for '{term.mention}'"
                )
            candidate_dicts = [candidate.as_dict() for candidate in candidates]
            candidate_index[term.placeholder_iri] = candidate_dicts
            # 동점 없는 100점 label 일치는 LLM 판단 없이 결정적으로 확정한다.
            exact_unique = candidates[0].score == 100 and (
                len(candidates) == 1 or candidates[1].score < 100
            )
            if exact_unique:
                selected = candidates[0]
                resolved[term.placeholder_iri] = ResolvedTerm(
                    placeholder_iri=term.placeholder_iri,
                    mention=term.mention,
                    selected_iri=selected.iri,
                    selected_label=selected.label,
                    method="exact",
                    score=selected.score,
                )
                continue
            pending.append(
                {
                    "placeholder_iri": term.placeholder_iri,
                    "mention": term.mention,
                    "expected_type": term.expected_type,
                    "candidates": candidate_dicts,
                }
            )

        # 모호한 placeholder를 한 번에 전달해 질문 전체 문맥에서 서로 일관되게 고른다.
        if pending:
            try:
                refinement = self.chain.invoke(
                    {
                        "question": question,
                        "candidates": json.dumps(pending, ensure_ascii=False, indent=2),
                    }
                )
            except (OutputParserException, ValueError) as error:
                raise TermResolutionError(
                    f"Term model returned an invalid result: {error}"
                ) from error
            if not isinstance(refinement, TermRefinement):
                raise TermResolutionError("Term model did not return a TermRefinement")

            # 누락·중복 placeholder가 있으면 부분 정합을 허용하지 않는다.
            choices = {choice.placeholder_iri: choice for choice in refinement.choices}
            expected = {str(item["placeholder_iri"]) for item in pending}
            if len(choices) != len(refinement.choices) or set(choices) != expected:
                raise TermResolutionError(
                    "Term model did not return each placeholder exactly once"
                )

            term_by_placeholder = {term.placeholder_iri: term for term in terms}
            for placeholder in sorted(expected):
                choice = choices[placeholder]
                mention = term_by_placeholder[placeholder].mention
                if choice.selected_iri is None:
                    raise TermResolutionError(f"No ontology term matches '{mention}'")
                # 모델 출력도 후보 allowlist와 점수 임계값을 다시 통과해야 한다.
                candidate = next(
                    (
                        item
                        for item in candidate_index[placeholder]
                        if item["iri"] == choice.selected_iri
                    ),
                    None,
                )
                if candidate is None:
                    raise TermResolutionError(
                        f"Term model selected an unknown IRI: {choice.selected_iri}"
                    )
                if (
                    float(candidate["score"]) < MIN_LLM_TERM_SCORE
                    or choice.confidence < MIN_LLM_TERM_CONFIDENCE
                ):
                    raise TermResolutionError(
                        f"Ontology term match was too weak for '{mention}'"
                    )
                resolved[placeholder] = ResolvedTerm(
                    placeholder_iri=placeholder,
                    mention=mention,
                    selected_iri=str(candidate["iri"]),
                    selected_label=str(candidate["label"]),
                    method="llm",
                    score=float(candidate["score"]),
                )

        # dict 삽입 순서가 아니라 원래 QueryDraft의 term 순서를 명시적으로 복원한다.
        return [resolved[term.placeholder_iri] for term in terms]
