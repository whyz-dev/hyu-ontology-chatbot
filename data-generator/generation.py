from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import Any


QUESTION_TYPES = [
    "direct_fact",
    "time_range",
    "deadline",
    "duration",
    "yes_no",
    "eligibility",
    "prerequisite",
    "procedure",
    "restriction",
    "exception",
    "min_max",
    "count",
    "comparison",
    "ordering",
    "list",
    "definition",
    "scope",
    "reason",
    "consequence",
    "alternative",
    "what_if",
    "condition_chain",
    "personal_scenario",
    "clarification",
    "contextual_followup",
    "user_correction",
    "confirmation",
    "multi_evidence",
    "cross_document",
    "negative_question",
]

SINGLE_EVIDENCE_QUESTION_TYPES = [
    item for item in QUESTION_TYPES if item not in {"multi_evidence", "cross_document"}
]

DIALOGUE_MODES = [
    "standalone",
    "clarification",
    "contextual_followup",
    "user_correction",
    "confirmation",
]

CONTEXTUAL_QUESTION_TYPES = set(DIALOGUE_MODES) - {"standalone"}
STANDALONE_QUESTION_TYPES = [
    item
    for item in SINGLE_EVIDENCE_QUESTION_TYPES
    if item not in CONTEXTUAL_QUESTION_TYPES
]


TYPE_GUIDE = """
- direct_fact: 단일 사실, time_range: 시작과 종료, deadline: 마감 시각
- duration: 기간 길이, yes_no: 예/아니오, eligibility: 대상 자격
- prerequisite: 선행 조건, procedure: 문서에 명시된 순서, restriction: 금지나 제한
- exception: 일반 규칙의 예외, min_max: 최소·최대 수치, count: 개수
- comparison: 두 조건 비교, ordering: 시간·절차의 선후관계, list: 둘 이상 열거
- definition: 제도 정의, scope: 적용 범위, reason: 문서에 명시된 이유
- consequence: 조건의 결과, alternative: 가능한 다른 방법, what_if: 가정 상황
- condition_chain: 둘 이상의 조건이 연쇄되는 질문, personal_scenario: 개인 상황
- clarification: 모호한 첫 질문에 챗봇이 필요한 조건을 묻고 사용자가 답함
- contextual_followup: 첫 문답 뒤 대상을 생략한 후속 질문
- user_correction: 사용자가 앞서 말한 조건을 명시적으로 정정함
- confirmation: 사용자가 이해한 내용을 다시 확인함
- multi_evidence: 서로 다른 두 페이지의 근거를 함께 사용함
- cross_document: 학사안내와 정책학과 공지를 함께 사용함
- negative_question: 안 되나요·못 하나요 같은 부정형 질문
""".strip()


def _message_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
    }


def _query_ir_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "subject": {"type": "string"},
            "constraints": {"type": "object"},
            "requested_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["intent", "subject", "constraints", "requested_fields"],
    }


def _bundle_schema(dialogue_mode: str) -> dict[str, Any]:
    message_count = 1 if dialogue_mode == "standalone" else 3
    allowed_question_types = (
        STANDALONE_QUESTION_TYPES
        if dialogue_mode == "standalone"
        else [dialogue_mode]
    )
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "question_type": {
                "type": "string",
                "enum": allowed_question_types,
            },
            "style_source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "messages": {
                "type": "array",
                "items": _message_schema(),
                "minItems": message_count,
                "maxItems": message_count,
            },
            "reference_answer": {"type": "string"},
            "query_ir": _query_ir_schema(),
        },
        "required": [
            "topic",
            "question_type",
            "style_source_ids",
            "messages",
            "reference_answer",
            "query_ir",
        ],
    }


def response_schema(dialogue_modes: list[str]) -> dict[str, Any]:
    bundle_keys = [f"bundle_{index:02d}" for index in range(1, len(dialogue_modes) + 1)]
    return {
        "type": "object",
        "properties": {
            "bundles": {
                "type": "object",
                "properties": {
                    key: _bundle_schema(dialogue_mode)
                    for key, dialogue_mode in zip(bundle_keys, dialogue_modes)
                },
                "required": bundle_keys,
            }
        },
        "required": ["bundles"],
    }


def build_prompt(
    style_questions: list[dict[str, str]],
    round_number: int,
    evidence_assignments: list[dict[str, str]],
    dialogue_modes: list[str],
    preferred_types: list[str],
) -> str:
    styles = "\n".join(
        f"- {item['source_id']}: {item['question']}" for item in style_questions
    )
    evidence = "\n".join(
        f"[bundle_{index:02d}] {item['source_unit_id']} / {item['locator']} / "
        f"대화 형식: {dialogue_modes[index - 1]}\n"
        f"{item.get('prompt_content', item['quote'])}"
        for index, item in enumerate(evidence_assignments, start=1)
    )
    bundle_count = len(evidence_assignments)
    return f"""당신은 한양대학교 2026학년도 2학기 수강신청 질문 데이터 생성기다.

이번 작업은 다양성 배치 {round_number}이다. 아래에서 각 bundle에 미리 배정한 근거만
사용하여 질문을 만든다.

아래 UVaBot 질문에서는 질문 길이, 생략, 구어체, 오탈자, 인사말, 개인 상황 설명 같은
표현 방식만 참고한다. Valladolid의 기관명, 사실, 날짜, 금액, 답변은 가져오지 않는다.

[UVaBot 질문 스타일]
{styles}

[bundle별 고정 한양대학교 근거]
{evidence}

[질문 유형]
{TYPE_GUIDE}

[이번 배치에서 우선 사용할 질문 유형]
{', '.join(preferred_types)}

[생성 지시]
1. bundle_01부터 bundle_{bundle_count:02d}까지 정확히 하나씩 만든다. 각 bundle은 자신에게
   배정된 근거 하나만 사용하며 다른 bundle의 근거를 가져오거나 합치지 않는다.
2. 각 bundle은 질문 하나만 가진다. 한 질문을 여러 말투로 바꾼 변형이나 같은 답을 확인하는
   후속 버전은 만들지 않는다. bundle끼리 query_ir와 묻는 사실이 겹치지 않아야 한다.
3. 배정 근거가 표 행이면 그 행의 열 관계를 유지한다. 근거에 여러 사실이 있으면 다른
   bundle과 겹치지 않는 가장 구체적인 사실 하나를 선택한다.
4. question_type은 질문 내용에 가장 잘 맞는 유형을 선택한다. 이번 배치 우선 유형을 가능한
   한 고르게 쓰되, 근거에 맞지 않는 유형을 억지로 붙이지 않는다. 단, 대화 형식이
   clarification, contextual_followup, user_correction, confirmation이면 question_type도
   해당 형식과 같은 값으로 쓰고, standalone에는 이 네 유형을 쓰지 않는다.
5. 각 bundle에 지정된 대화 형식을 정확히 따른다. standalone은 user 메시지 하나만 쓴다.
   나머지는 user-assistant-user 세 메시지를 쓰며 마지막 user가 평가 대상 질문이다.
   - clarification: 첫 질문에서 조건 하나를 생략하고, assistant가 그 조건을 물은 뒤 사용자가 답한다.
   - contextual_followup: 앞 문답에서 대상을 제시하고, 마지막 질문은 그 대상을 자연스럽게 생략한다.
   - user_correction: 사용자가 처음 말한 조건을 마지막 메시지에서 명시적으로 바로잡는다.
   - confirmation: assistant가 근거 내용을 짧게 설명하고 사용자가 구체적 해석을 확인한다.
   중간 assistant도 배정 근거 안의 정보만 사용하고 최종 reference_answer를 미리 그대로 말하지 않는다.
6. 자연스러운 한국어를 사용한다. 짧은 명사구, 정중한 질문, 오탈자 한두 개가 있는 질문,
   긴 개인 상황, 조건 생략, 부정형, 확인형을 골고루 섞는다.
7. reference_answer는 해당 질문에 답하는 배정 근거의 연속 문자열을 글자·띄어쓰기·문장부호
   그대로 복사한다. 조사나 종결어미도 고치지 말고, 근거에 없는 '네', '아니요', 설명을
   앞뒤에 붙이지 않는다. 표라면 답에 필요한 셀 하나 또는 연속된 셀 내용만 복사한다.
8. 근거 문장을 출력 JSON에 다시 작성하지 않는다. reference_answer의 날짜·시각·숫자·
   대상·예외는 배정 근거의 표기 그대로 사용한다. 24시는 밤 12시로 바꾸지 않는다.
9. OCR이 깨져 의미가 불확실한 부분은 추측하지 않는다. 24시간제 시각에 근거 없이 오전·오후를
   붙이지 않는다. 한 bundle 안에서 모순되는 내용을 만들지 않는다.
10. 이번 데이터는 bundle마다 근거 하나만 사용하므로 multi_evidence와 cross_document
    유형은 사용하지 않는다.
11. '검수 필요', '자동 승인', '상태', OCR 정확도, 마크다운 기호는 파서의 검수 정보이므로
    질문·답·query_ir의 대상으로 절대 사용하지 않는다.
12. query_ir은 IRI나 SPARQL 없이 intent, subject, constraints, requested_fields만 쓰고
    질문과 대화에 누적된 조건을 모두 반영한다.
13. style_source_ids에는 위 UVaBot ID 중 실제 참고한 것을 1~2개 기록한다.
14. 이전 배치에서 흔히 나올 법한 '수강신청 언제예요?'만 반복하지 말고, 이번 주제의
    세부 조건·예외·비교·상황을 적극적으로 사용한다.
"""


def _post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error


def model_digest(base_url: str, model: str) -> str:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as result:
        tags = json.loads(result.read().decode("utf-8"))
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return str(item.get("digest", ""))
    raise RuntimeError(f"Ollama model is not installed: {model}")


def generate_round(
    base_url: str,
    model: str,
    prompt: str,
    num_ctx: int,
    round_number: int,
    dialogue_modes: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": response_schema(dialogue_modes),
        "options": {
            "temperature": 0.3,
            "seed": 41 + round_number,
            "num_ctx": num_ctx,
            "num_predict": 18000,
        },
        "keep_alive": "10m",
    }
    response = _post_json(f"{base_url}/api/chat", payload)
    content = response.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty structured response")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        reason = response.get("done_reason", "unknown")
        raise RuntimeError(
            f"Ollama returned invalid JSON ({len(content)} characters, "
            f"done_reason={reason}): {error}"
        ) from error
    metadata = {
        "model": model,
        "model_digest": model_digest(base_url, model),
        "round": round_number,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get("total_duration"),
    }
    return parsed["bundles"], metadata


def flatten_bundles(
    bundles: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    evidence_assignments: list[dict[str, str]],
    dialogue_modes: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (bundle_key, bundle) in enumerate(bundles.items()):
        expected_key = f"bundle_{index + 1:02d}"
        if bundle_key != expected_key:
            raise ValueError(f"Unexpected bundle order: {bundle_key}, expected {expected_key}")
        answer = str(bundle["reference_answer"]).strip()
        messages = [dict(message) for message in bundle["messages"]]
        expected_roles = (
            ["user"]
            if dialogue_modes[index] == "standalone"
            else ["user", "assistant", "user"]
        )
        for message, role in zip(messages, expected_roles):
            message["role"] = role
        if dialogue_modes[index] == "clarification" and len(messages) >= 2:
            messages[1]["content"] = "어떤 조건을 기준으로 확인할까요?"
        elif dialogue_modes[index] != "standalone" and len(messages) >= 2:
            messages[1]["content"] = answer
        if not messages or messages[0].get("role") != "user":
            messages.insert(0, {"role": "user", "content": ""})
        if messages[-1].get("role") == "assistant":
            messages[-1]["content"] = answer
        else:
            messages.append({"role": "assistant", "content": answer})
        records.append(
            {
                "topic": bundle["topic"],
                "question_type": bundle["question_type"],
                "dialogue_mode": dialogue_modes[index],
                "style_source_ids": bundle["style_source_ids"],
                "messages": messages,
                "query_ir": bundle["query_ir"],
                "reference_answer": answer,
                "evidence": [
                    {
                        "source_unit_id": evidence_assignments[index]["source_unit_id"],
                        "locator": evidence_assignments[index]["locator"],
                        "quote": evidence_assignments[index]["quote"],
                    }
                ],
                "_generation": metadata,
            }
        )
    return records
