"""View의 질문을 Model과 LLM 처리 단계에 연결하는 Controller 계층.

Controller는 질의 생성, 용어 정합, SPARQL 실행, 답변 생성을 순서대로 조정한다.
RDF 저장·조회 구현은 Model에, 화면 출력과 세션 상태는 View에 남긴다.
"""
