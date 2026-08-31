"""실행 설정 입력과 명시적인 초기화 이벤트를 담당하는 사이드바 View."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from models.settings import SETTINGS


def render_sidebar(clear_chatbot_cache: Callable[[], None]) -> tuple[str, str, str]:
    """현재 설정값을 반환하고 캐시·대화 초기화 버튼을 처리한다.

    캐시 구현을 직접 참조하지 않고 상위 View가 전달한 callback을 실행한다. 이 경계로
    인해 사이드바는 Controller 생성 방식과 무관하게 사용자 이벤트만 담당한다.
    """
    with st.sidebar:
        st.header("실행 설정")
        ontology_path = st.text_input(
            "온톨로지 디렉터리 또는 TTL",
            value=str(SETTINGS.ontology_path),
        )
        model = st.text_input("Ollama 모델", value=SETTINGS.model)
        ollama_url = st.text_input("Ollama URL", value=SETTINGS.ollama_url)

        if st.button("온톨로지 다시 불러오기", width="stretch"):
            # 설정별 Controller/RDF graph 캐시 폐기는 상위 View에 위임한다.
            clear_chatbot_cache()
            st.success("챗봇 캐시를 비웠습니다.")
        if st.button("대화 초기화", width="stretch"):
            # Streamlit 재실행 모델에서 빈 기록이 즉시 반영되도록 rerun한다.
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("모델은 자동으로 다운로드하지 않습니다.")
        st.caption("대화 기록은 현재 브라우저 세션에만 저장됩니다.")
    return ontology_path.strip(), model.strip(), ollama_url.strip()
