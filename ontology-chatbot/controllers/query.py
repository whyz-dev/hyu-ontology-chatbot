"""자연어 질문을 ontology schema에 맞는 안전한 SPARQL 초안으로 변환한다.

Qwen은 resource IRI 대신 정합 가능한 placeholder를 포함한 ``QueryDraft``를 만든다.
Controller는 생성 결과를 실행하지 않고, 금지된 IRI 추측·어휘·질의 형태를 검증한 뒤
Model이 실행할 수 있는 읽기 전용 SELECT로 보정한다.
"""

from __future__ import annotations

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from models.ontology import OntologyStore
from models.schemas import QueryDraft, QueryGenerationError
from models.settings import SETTINGS

from .prompts import QUERY_PROMPT, query_prompt_messages


# RDFLib은 RDFS 추론을 자동 적용하지 않는다. LLM이 추상 상위 class를 직접 type으로
# 사용했을 때는 같은 의미의 명시적 class 경로로만 보수적으로 교정한다. 전체 IRI를 써서
# 모델이 rdf/rdfs PREFIX를 빠뜨렸더라도 교정 결과가 독립적으로 파싱되게 한다.
_ACADEMIC_RULE_TYPE_PATTERN = re.compile(
    r"(?P<subject>[?$][A-Za-z_][A-Za-z0-9_-]*)"
    r"(?P<before>\s+)(?:a|rdf:type)(?P<after>\s+)hyu:AcademicRule\b",
    flags=re.IGNORECASE,
)
_ACADEMIC_RULE_TYPE_PATH = (
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>/"
    "<http://www.w3.org/2000/01/rdf-schema#subClassOf>*"
)


def reconcile_query(sparql: str) -> str:
    """안전하게 의미를 보존할 수 있는 알려진 LLM 질의 오류만 교정한다.

    ``?rule a hyu:AcademicRule``은 명시된 하위 class 인스턴스를 찾지 못하는 추상 type
    조회다. 이를 ``rdf:type/rdfs:subClassOf*`` property path로 바꾸면 추론 없이도 같은
    class 계층 의미를 유지할 수 있다. 그 밖의 vocabulary·placeholder·문법 오류는 손대지
    않고 후속 검증기가 거부하도록 둔다.
    """

    return _ACADEMIC_RULE_TYPE_PATTERN.sub(
        lambda match: (
            f"{match.group('subject')}{match.group('before')}"
            f"{_ACADEMIC_RULE_TYPE_PATH}{match.group('after')}hyu:AcademicRule"
        ),
        sparql,
    )


class QueryGenerator:
    """질의 생성 LLM과 결정적 SPARQL 검증을 결합한 Controller.

    질문과 현재 ontology schema를 입력받아 ``QueryDraft``와 안전하게 정규화된 SELECT를
    반환한다. 여기서는 질의를 실행하거나 검색 결과를 만들지 않는다.
    """

    def __init__(self, store: OntologyStore, *, model: str, ollama_url: str) -> None:
        """온톨로지 저장소와 Qwen의 JSON Schema 출력 chain을 구성한다."""

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
            query_prompt_messages()
        ) | llm.with_structured_output(QueryDraft, method="json_schema")

    def validate(self, draft: QueryDraft) -> str:
        """LLM 초안을 검사하고 실행 가능한 읽기 전용 SELECT를 반환한다.

        resource IRI와 label을 직접 추측하는 질의, placeholder 선언 불일치, ontology에
        없는 vocabulary를 거부한다. 실패 시 ``ValueError``를 발생시켜 ``create``의 제한된
        재시도 대상으로 돌린다.
        """

        # 알려진 추상 class 직접 조회는 안전한 명시적 상속 경로로 먼저 정합한다.
        # 교정된 질의도 아래의 모든 기존 안전성·vocabulary·문법 검사를 다시 통과해야 한다.
        sparql = reconcile_query(draft.sparql)
        # 개체 IRI는 질문 표현을 실제 ontology 후보에 정합한 뒤에만 들어갈 수 있다.
        if re.search(
            r"(?:\bres:|<urn:hyu-chatbot:resource:)",
            sparql,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Query guessed a resource IRI instead of using a placeholder"
            )
        # label 문자열 비교는 별도의 용어 정합 단계를 우회하므로 허용하지 않는다.
        if re.search(
            r"(?:rdfs:label|skos:(?:prefLabel|altLabel)|schema:name)\s+[\"']",
            sparql,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Query compared a resource label directly instead of using a placeholder"
            )
        # RDFLib은 RDFS 추론을 자동 수행하지 않으므로 추상 상위 class만 조회하는 흔한
        # 오류를 명시적으로 막는다.
        if re.search(
            r"(?:\ba|rdf:type)\s+hyu:AcademicRule\b",
            sparql,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Do not query '?rule a hyu:AcademicRule' without an explicit "
                "rdfs:subClassOf* path"
            )

        # SPARQL의 placeholder와 구조화 terms는 누락이나 잉여 없이 일대일이어야 한다.
        declared = {term.placeholder_iri for term in draft.terms}
        used = set(re.findall(r"urn:query-term:[A-Za-z0-9_-]+", sparql))
        if declared != used:
            missing = sorted(used - declared)
            unused = sorted(declared - used)
            raise ValueError(
                f"Placeholder mismatch: undeclared={missing}, unused={unused}"
            )

        # Model이 실제 graph vocabulary와 SELECT 안전 규칙을 최종 확인하고 제한을 보정한다.
        self.store.validate_query_vocabulary(sparql)
        return self.store.safe_select_query(sparql)

    def create(self, question: str) -> tuple[QueryDraft, str, int]:
        """질문으로 질의 초안을 만들고 검증된 SPARQL과 호출 횟수를 반환한다.

        Qwen에는 질문, graph schema, 고정 guidance와 이전 검증 오류를 제공한다. 구조화
        출력 또는 안전성 검증이 실패하면 이전 초안과 오류만 전달해 한 번 교정하며, 다른
        검색 방식으로 fallback하지 않는다. 두 번째 실패는 ``QueryGenerationError``다.
        """

        feedback = "첫 시도입니다. 이전 오류가 없습니다."
        previous: QueryDraft | None = None
        for attempt in range(1, 3):
            draft: QueryDraft | None = None
            try:
                # with_structured_output이 자유 형식 설명 대신 QueryDraft만 반환하게 한다.
                draft = self.chain.invoke(
                    {
                        "question": question,
                        "schema": self.store.schema_catalog(),
                        "guidance": QUERY_PROMPT.guidance_text,
                        "feedback": feedback,
                    }
                )
                if not isinstance(draft, QueryDraft):
                    raise ValueError(  # noqa: TRY004 - 재시도할 출력 검증 오류다.
                        "Query model did not return a QueryDraft"
                    )
                return draft, self.validate(draft), attempt
            except (OutputParserException, ValueError) as error:
                if draft is not None:
                    previous = draft
                if attempt == 2:
                    raise QueryGenerationError(str(error), draft or previous) from error
                # 온톨로지나 prompt를 바꾸지 않고 직전 초안의 검증 오류만 수정하게 한다.
                feedback = json.dumps(
                    {
                        "error": str(error),
                        "previous_draft": (
                            previous.model_dump() if previous is not None else None
                        ),
                        "instruction": (
                            "오류만 수정해 완전한 QueryDraft를 다시 작성하세요. "
                            "다른 자료를 검색하거나 resource IRI를 추측하지 마세요."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        raise RuntimeError("Query generation ended unexpectedly")
