"""Streamlit 세션, 화면 구성 요소, 온톨로지 Controller를 조정하는 View 진입점.

Streamlit은 사용자 이벤트마다 파일을 위에서부터 다시 실행한다. 이 모듈은 대화 기록을
``session_state``에 유지하고, 설정별 Controller를 캐시한 뒤, 하나의 질문 이벤트를
Controller에 전달하여 반환된 trace를 화면에 그리는 전체 흐름을 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from controllers.chatbot import OntologyQA

from .chat import render_example_picker, render_message
from .sidebar import render_sidebar


@st.cache_resource(show_spinner=False)
def _load_chatbot(
    ontology_path: str,
    model: str,
    ollama_url: str,
) -> OntologyQA:
    """읽기 전용 RDF graph와 Qwen client를 설정 조합별로 한 번만 만든다.

    Streamlit 재실행 때마다 큰 온톨로지를 다시 읽지 않게 하는 View 수준의 수명 관리다.
    질문 해석과 검색 자체는 반환되는 :class:`OntologyQA` Controller가 수행한다.
    """
    return OntologyQA(
        Path(ontology_path).expanduser(),
        model=model,
        ollama_url=ollama_url,
    )


def _runtime_error(error: Exception) -> dict[str, object]:
    """View 경계 밖으로 전파된 예외를 일반 assistant 메시지 형태로 바꾼다."""
    return {
        "role": "assistant",
        "content": (
            "답변 생성 중 오류가 발생했습니다. 실행 설정과 Ollama 상태를 확인하세요."
        ),
        "trace": {
            "status": "runtime_error",
            "query_attempts": 0,
            "result": {"rows": [], "row_count": 0},
            "runtime_error": f"{type(error).__name__}: {error}",
        },
    }


def main() -> None:
    """화면을 구성하고 현재 실행에서 발생한 질문 이벤트 하나를 처리한다.

    실행 순서는 설정 수집, 세션 복원, 기존 메시지 재렌더링, 새 입력 수집,
    Controller 호출, 응답 저장이다. 기존 메시지를 먼저 다시 그리는 것은 매 이벤트마다
    전체 스크립트를 재실행하는 Streamlit의 동작 방식에 따른 것이다.
    """
    st.set_page_config(
        page_title="HYU 수강신청 온톨로지 챗봇",
        page_icon="🎓",
        layout="wide",
    )
    st.title("🎓 HYU 수강신청 온톨로지 챗봇")
    st.caption(
        "2026학년도 2학기 수강편람 수동 온톨로지를 검색해 답합니다. "
        "대화는 화면에 이어지지만 현재 버전은 각 질문을 독립적으로 해석합니다."
    )

    ontology_path, model, ollama_url = render_sidebar(_load_chatbot.clear)

    # 브라우저 세션이 유지되는 동안 이전 대화를 다시 그릴 수 있도록 View 상태만 보관한다.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    selected_example = None
    if not st.session_state.messages:
        selected_example = render_example_picker()

    # 예시 버튼과 채팅 입력을 같은 질문 이벤트로 합쳐 이후 처리 경로를 하나로 유지한다.
    question = st.chat_input("2026-2 수강신청에 관해 질문하세요") or selected_example
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.spinner("질문을 SPARQL로 바꾸고 온톨로지를 검색하는 중입니다..."):
            # View/Controller 경계: 질문을 넘기고 화면에 필요한 trace만 돌려받는다.
            chatbot = _load_chatbot(ontology_path, model, ollama_url)
            trace = chatbot.ask(question)
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": str(trace["answer"]),
            "trace": trace,
        }
    except Exception as error:  # noqa: BLE001 - 한 실패가 앱 전체를 막지 않는다.
        # 예외도 메시지로 보존해 다음 Streamlit 재실행에서 대화 흐름이 끊기지 않게 한다.
        assistant_message = _runtime_error(error)

    render_message(assistant_message)
    st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
