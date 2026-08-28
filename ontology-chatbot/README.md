# Ontology chatbot

Qwen이 한국어 질문을 SPARQL로 바꾸고, 수동 온톨로지에서 검색한 결과만 사용해 답한다.
LangChain의 `ChatOllama`와 Ollama native JSON Schema 출력을 사용한다.

## 흐름

```text
질문
→ Qwen SPARQL 초안과 resource placeholder 생성
→ ontology label 후보 검색 및 Qwen 용어 정합
→ RDFLib으로 읽기 전용 SELECT 실행
→ 검색 행과 원문 근거를 Qwen에 전달
→ 근거 기반 한국어 답변
```

Qwen은 ontology resource IRI를 직접 만들지 않는다. 초안의
`<urn:query-term:...>`만 `skos:prefLabel`, `skos:altLabel`, `rdfs:label`,
`schema:name`에서 검색된 기존 IRI로 바꾼다. 정확히 하나의 label과 일치하면 코드가
확정하고, 모호한 후보만 Qwen이 고른다.

질문마다 정합된 SPARQL을 정확히 한 번 실행한다. 질의 생성 실패나 용어 정합 실패,
0행 결과를 원문 문자열 검색으로 우회하지 않는다. 따라서 반환된 상태는 실제로 실패한
단계를 그대로 나타내며, 관련 없는 `sourceText`가 답변 근거로 섞이지 않는다.

질문의 핵심 표현을 기존 ontology resource에 안전하게 연결하지 못하면
`term_unresolved`로 종료한다. 이 상태와 `query_generation_failed`,
`query_execution_failed`, 실행 결과가 0행인 `no_result`, 결과의 답변 변환에 실패한
`answer_generation_failed`를 서로 구분한다. 어느 실패도 실제 세계에 정보가 없다는
뜻으로 해석하지 않는다.

SPARQL은 로컬 `SELECT`만 허용한다. mutation, 원격 `SERVICE`, 외부 `FROM`은 거부하고
바깥 질의 결과를 최대 50행으로 제한한다. 형식 검증 실패에는 같은 생성 단계를 한 번만
재시도하며, 다른 검색 방식이나 자료로 대체하지 않는다.

## 코드 구조

```text
ontology-chatbot/
├── app.py                  Streamlit 진입점
├── config.ini              온톨로지·Ollama·생성 설정
├── models/
│   ├── schemas.py          질의·정합·답변 데이터 계약
│   ├── settings.py         config.ini 검증과 설정 로딩
│   ├── ontology.py         RDF 로딩·schema·용어 후보 색인
│   └── sparql.py           안전 검사·치환·SELECT 실행
├── controllers/
│   ├── chatbot.py          전체 요청 흐름과 trace
│   ├── query.py            SPARQL 생성과 검증
│   ├── terms.py            ontology 용어 정합
│   ├── answer.py           근거 기반 답변 생성
│   └── prompts.py          prompt JSON·few-shot 검증과 조합
├── views/
│   ├── chat.py             메시지·근거·예시 질문 렌더링
│   ├── sidebar.py          실행 설정과 세션 동작
│   └── streamlit_app.py    Streamlit 화면 흐름
├── example.json              QueryDraft few-shot 데이터
├── prompts/
│   ├── query.json          SPARQL 생성 프롬프트
│   ├── terms.json          용어 정합 프롬프트
│   └── answer.json         근거 기반 답변 프롬프트
```

`app.py`는 Streamlit View를 연결하는 얇은 진입점이다. 내부 의존 방향은
`Views → Controllers → Models`이며, Controller만 LLM 프롬프트를 사용한다. Model은
RDF 데이터와 SPARQL 실행을, View는 입출력만 담당한다.

Streamlit은 입력 때마다 화면 스크립트를 다시 실행하므로 View 상태를 별도 class instance에
두지 않는다. 대화 상태는 `st.session_state`, 무거운 챗봇 객체는 `st.cache_resource`가
관리하고 각 View 모듈은 작은 렌더 함수로 구성한다.

프롬프트 본문은 `prompts/*.json`, few-shot 데이터는 `example.json`에서 관리한다.
`controllers/prompts.py`가 JSON schema, QueryDraft 구조와 중복 ID를 검증한 뒤 LangChain
메시지로 조합한다. `prompts/`에는 실행 가능한 Python 코드를 두지 않는다.

기본 온톨로지 경로, Ollama 모델·URL·seed와 단계별 생성 token 수는 `config.ini`에서
관리한다. 설정 section이나 key가 누락되거나 예상하지 않은 key가 있으면 시작 시 즉시
오류를 내며 코드 내부 기본값으로 우회하지 않는다. Streamlit 사이드바에서는 현재 실행에
한해 온톨로지 경로, 모델과 URL을 바꿀 수 있다.

## 실행

먼저 로컬에 다음 파일과 모델이 있어야 한다.

```text
data/ontology/published/
└── hyu-course-guide-2026-2.ttl
```

```bash
uv sync
ollama pull qwen3.5:9b

uv run streamlit run ontology-chatbot/app.py
```

브라우저에서 질문을 입력하면 답변이 대화 기록에 쌓인다. 답변 아래의
`근거와 온톨로지 질의 보기`를 펼치면 정합된 용어, 실제 실행한 SPARQL, 검색 결과 행을
확인할 수 있다. 사이드바에서는 TTL 경로, Ollama 모델과 URL을 바꾸거나 대화 기록과
챗봇 캐시를 초기화할 수 있다.

현재 UI는 대화 내역을 세션에 보관하지만, 각 질문은 이전 발화 없이 독립적으로
온톨로지 질의로 변환한다.

## 제한

- RDFLib은 RDFS/OWL 추론을 자동 적용하지 않는다. prompt가 필요한 class path를 명시한다.
- `hasException`은 규칙 엔진처럼 자동 계산되지 않으므로 검색 결과에 포함시켜 답변 단계에서
  조건을 설명한다.
- 그래프에 명시되지 않은 허용·금지는 `false`가 아니라 알 수 없음으로 처리한다.
- 수동 온톨로지의 원문 충돌과 입학연도 구간 누락은 별도 데이터 검수 대상이다.
