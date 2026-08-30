"""채팅 본문과 온톨로지 검색 trace를 그리는 순수 View 구성 요소.

이 모듈은 Controller를 직접 호출하거나 세션을 변경하지 않는다. 전달받은 메시지와
trace를 사람이 읽을 수 있는 Streamlit 요소로 바꾸는 렌더링 책임만 가진다.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

EXAMPLE_QUESTIONS = (
    "25학번 건축공학과의 졸업 학점은 몇 점이야?",
    "3학년 모의수강신청은 언제야?",
    "60점 미만이면 성적은 어떻게 처리돼?",
)


def _binding_value(value: object) -> object:
    """RDFLib binding JSON을 사람이 읽기 쉬운 셀 값으로 줄인다."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _compact_rows(trace: dict[str, Any]) -> list[dict[str, object]]:
    """Controller trace의 RDFLib 결과 행을 표 렌더링용 목록으로 변환한다."""
    result = trace.get("result")
    if not isinstance(result, dict):
        return []
    rows = result.get("rows")
    if not isinstance(rows, list):
        return []
    return [
        {str(key): _binding_value(value) for key, value in row.items()}
        for row in rows
        if isinstance(row, dict)
    ]


def _render_trace(trace: dict[str, Any]) -> None:
    """질문 처리 trace를 상태 요약·질의·결과의 세 영역으로 표시한다.

    Controller가 기록한 중간 결과를 그대로 설명하되, 이곳에서 질의를 고치거나
    재실행하지는 않는다. 따라서 화면 표시와 추론 파이프라인의 경계가 유지된다.
    """
    with st.expander("근거와 온톨로지 질의 보기"):
        result = trace.get("result")
        row_count = result.get("row_count", 0) if isinstance(result, dict) else 0
        status_label = {
            "answered": "답변 완료",
            "no_result": "근거 부족",
            "query_generation_failed": "질의 생성 실패",
            "term_unresolved": "용어 정합 실패",
            "query_execution_failed": "질의 실행 실패",
            "answer_generation_failed": "답변 생성 실패",
            "runtime_error": "실행 오류",
        }.get(str(trace.get("status")), str(trace.get("status", "알 수 없음")))

        first, second, third = st.columns(3)
        first.metric("상태", status_label)
        second.metric("질의 생성 시도", int(trace.get("query_attempts", 0)))
        third.metric("검색 행", int(row_count))

        summary_tab, query_tab, result_tab = st.tabs(
            ["용어 정합", "SPARQL", "검색 결과"]
        )
        with summary_tab:
            terms = trace.get("resolved_terms")
            if isinstance(terms, list) and terms:
                st.markdown("**정합된 온톨로지 용어**")
                st.dataframe(
                    [
                        {
                            "질문 표현": term.get("mention"),
                            "선택 용어": term.get("selected_label"),
                            "IRI": term.get("selected_iri"),
                            "방법": term.get("method"),
                            "점수": term.get("score"),
                        }
                        for term in terms
                        if isinstance(term, dict)
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("정합할 개체 용어가 없거나 정합 단계 전에 종료되었습니다.")

            for key, label in (
                ("query_generation_error", "질의 생성 오류"),
                ("term_resolution_error", "용어 정합 오류"),
                ("query_execution_error", "질의 실행 오류"),
                ("answer_generation_error", "답변 생성 오류"),
                ("runtime_error", "실행 오류"),
            ):
                if trace.get(key):
                    st.error(f"{label}: {trace[key]}")

        with query_tab:
            query = trace.get("query")
            if query:
                st.markdown("**실행한 질의**")
                st.code(str(query), language="sparql", wrap_lines=True)
            else:
                st.info("실행된 SPARQL 질의가 없습니다.")

            draft_query = trace.get("draft_query")
            if draft_query and draft_query != query:
                st.markdown("**Qwen이 만든 정합 전 질의 초안**")
                st.code(str(draft_query), language="sparql", wrap_lines=True)

        with result_tab:
            rows = _compact_rows(trace)
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("검색된 온톨로지 행이 없습니다.")


def render_message(message: dict[str, Any]) -> None:
    """사용자 또는 assistant 메시지와 선택적 trace를 한 채팅 말풍선에 표시한다.

    사용자 메시지는 본문만 출력한다. assistant 메시지는 Controller 상태에 따라
    성공·근거 부족·오류 표현을 구분하고, 재현 가능한 검색 trace를 함께 제공한다.
    """
    role = str(message.get("role", "assistant"))
    with st.chat_message(role):
        content = str(message.get("content", ""))
        trace = message.get("trace")
        if role != "assistant" or not isinstance(trace, dict):
            st.markdown(content)
            return

        status = trace.get("status")
        if status == "answered":
            st.markdown(content)
        elif status == "no_result":
            st.warning(content)
        else:
            st.error(content)
        _render_trace(trace)


def render_example_picker() -> str | None:
    """예시 질문을 표시하고 클릭된 질문만 입력 이벤트로 반환한다.

    표시 시점은 상위 View가 결정한다. 이 함수는 대화 기록을 직접 추가하지 않으므로
    예시 버튼과 일반 채팅 입력이 동일한 처리 흐름을 사용할 수 있다.
    """
    st.markdown("**예시 질문**")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, question in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(question, width="stretch"):
            return question
    return None
