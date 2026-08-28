"""외부 JSON 프롬프트와 few-shot 예제를 LLM 메시지로 조합한다.

프롬프트 문구는 코드와 분리하되, Controller가 파일 schema와 template 변수를 시작 시
엄격하게 검증한다. 질의 few-shot은 ``QueryDraft`` 계약으로 읽어 실제 구조화 출력과 같은
형태로 Qwen에 제공한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Formatter
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from models.schemas import QueryDraft
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CHATBOT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = CHATBOT_ROOT / "prompts"
QUERY_EXAMPLES_PATH = CHATBOT_ROOT / "example.json"

PromptName = Literal["query", "terms", "answer"]

EXPECTED_HUMAN_VARIABLES = {
    "query": {"question", "schema", "guidance", "feedback"},
    "terms": {"question", "candidates"},
    "answer": {"question", "query", "results", "feedback"},
}


def _variables(template: str) -> set[str]:
    """Python format 문자열에서 LangChain이 채워야 할 변수 이름만 추출한다."""

    return {
        name
        for _, name, _, _ in Formatter().parse(template)
        if name is not None and name
    }


class PromptDocument(BaseModel):
    """역할별 system/human 문구와 선택적 guidance를 표현하는 JSON 계약."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["chat-prompt-v1"]
    name: PromptName
    system: list[str] = Field(min_length=1)
    human: list[str] = Field(min_length=1)
    guidance: list[str] | None = None

    @model_validator(mode="after")
    def guidance_matches_prompt(self) -> PromptDocument:
        """프롬프트 역할별 허용 필드와 template 변수 집합을 검증한다."""

        if self.name == "query" and not self.guidance:
            raise ValueError("The query prompt requires graph guidance")
        if self.name != "query" and self.guidance is not None:
            raise ValueError("Only the query prompt may define graph guidance")
        if not "\n".join(self.system).strip() or not "\n".join(self.human).strip():
            raise ValueError("Prompt text must not be empty")
        if _variables(self.system_text) or _variables(self.guidance_text):
            raise ValueError("Only the human prompt may contain template variables")
        variables = _variables(self.human_text)
        expected = EXPECTED_HUMAN_VARIABLES[self.name]
        if variables != expected:
            raise ValueError(
                f"Prompt variables do not match: expected={sorted(expected)}, "
                f"actual={sorted(variables)}"
            )
        return self

    @property
    def system_text(self) -> str:
        """JSON 줄 배열을 LangChain system 메시지 한 문자열로 복원한다."""

        return "\n".join(self.system)

    @property
    def human_text(self) -> str:
        """실행 시 변수가 치환될 human 메시지 template을 반환한다."""

        return "\n".join(self.human)

    @property
    def guidance_text(self) -> str:
        """질의 프롬프트에만 존재하는 graph 안내문을 문자열로 반환한다."""

        return "\n".join(self.guidance or [])


class QueryExample(BaseModel):
    """사용자 질문과 기대 QueryDraft를 묶은 질의 생성 few-shot 한 건."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    draft: QueryDraft


class QueryExampleFile(BaseModel):
    """루트 ``example.json``의 버전과 예제 목록을 고정하는 파일 계약."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["query-few-shot-v2"]
    examples: list[QueryExample] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> QueryExampleFile:
        """중복 예제가 prompt 편향을 만들지 않도록 ID와 질문을 고유하게 유지한다."""

        values = {
            "example_id": [example.example_id for example in self.examples],
            "question": [example.question for example in self.examples],
        }
        for name, items in values.items():
            if len(items) != len(set(items)):
                raise ValueError(f"Query few-shot {name} values must be unique")
        return self


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """JSON 파일을 지정된 Pydantic 계약으로 읽고 경로가 드러나는 오류를 만든다."""

    if not path.is_file():
        raise FileNotFoundError(f"Chatbot JSON file is missing: {path}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"Invalid chatbot JSON: {path}: {error}") from error


def load_prompt(name: PromptName) -> PromptDocument:
    """파일명과 JSON 내부 역할이 일치하는 prompt만 반환한다."""

    path = PROMPT_ROOT / f"{name}.json"
    document = _read_model(path, PromptDocument)
    if document.name != name:
        raise ValueError(
            f"Prompt name mismatch at {path}: expected={name}, actual={document.name}"
        )
    return document


def load_query_examples(path: Path = QUERY_EXAMPLES_PATH) -> tuple[QueryExample, ...]:
    """고정 JSON schema를 만족하는 QueryDraft 예제를 불변 순서로 반환한다."""

    payload = _read_model(path, QueryExampleFile)
    return tuple(payload.examples)


# 프롬프트 오류는 첫 질문 도중이 아니라 Controller import 시 즉시 드러나게 한다.
QUERY_PROMPT = load_prompt("query")
TERMS_PROMPT = load_prompt("terms")
ANSWER_PROMPT = load_prompt("answer")
QUERY_EXAMPLES = load_query_examples()


def query_prompt_messages() -> list[BaseMessage | tuple[str, str]]:
    """system → 고정 human/AI 예제 → 현재 질문 template 순서로 메시지를 만든다.

    반환된 마지막 human 메시지만 실행 시 ``question``, ``schema``, ``guidance``,
    ``feedback``으로 치환된다. 앞선 few-shot은 검수된 고정 메시지로 유지한다.
    """

    # BaseMessage를 사용해야 예제 SPARQL의 중괄호를 template 변수로 보지 않는다.
    messages: list[BaseMessage | tuple[str, str]] = [
        SystemMessage(content=QUERY_PROMPT.system_text)
    ]
    for example in QUERY_EXAMPLES:
        messages.extend(
            [
                HumanMessage(content=f"질문:\n{example.question}"),
                AIMessage(
                    content=json.dumps(
                        example.draft.model_dump(), ensure_ascii=False, indent=2
                    )
                ),
            ]
        )
    messages.append(("human", QUERY_PROMPT.human_text))
    return messages
