# HYU Ontology Chatbot

한양대학교의 공식 행정 문서를 온톨로지로 변환하고, 학생 질문을 온톨로지 질의로
해석하는 챗봇 연구 프로젝트다.

첫 번째 vertical slice는 **2026학년도 2학기 수강신청**이다. 작은 공식 문서 집합으로
수집부터 질의까지 검증한 뒤 범위를 넓힌다.

## 첫 범위

- 지식 범위: 2026학년도 2학기 수강신청
- 지식 원천: 한양대학교 공식 공지, 안내문, 수강편람, 학사일정
- 질문 출력: 한국어 질문과 ontology-independent Query IR
- 지식 원칙: 적용 학기·캠퍼스·학생 유형·게시 시각·수집 시각·원문 해시 보존
- 제외: GraphRAG, 개인 HY-in 상태, 잔여석처럼 인증이 필요한 실시간 정보

## 패키지

- [`data-parser/`](data-parser/README.md): 공식 문서 수집, OCR, 표 구조 복원과 검증
- [`data-generator/`](data-generator/README.md): 외부 질문 표현과 HYU 근거를 결합한 QA 후보 생성
- [`population-tool/`](population-tool/README.md): 문서 단위 ontology 초안 생성과 통합 검수

## 브랜치 규칙

각 패키지는 최신 `main`에서 `feat/<패키지 이름>`으로 분기한다. 기능 검증 후
`main`에 merge하고, 다음 패키지는 갱신된 `main`에서 다시 분기한다. 아직 검증이 끝나지
않은 패키지는 feature branch에만 유지한다.

현재 `main`에는 `data-parser` → `data-generator` → `population-tool` 순서로 merge되어
있다. `ontology-chatbot`은 `feat/ontology-chatbot`에서 개발하며 아직 `main`에 포함하지
않는다.

## 개발

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync
```

## 데이터

수동 전사본과 통합 온톨로지는 Hugging Face
[`whyz-dev/hyu-ontology`](https://huggingface.co/whyz-dev/hyu-ontology/tree/v2.0)의
`v2.0` 브랜치에서 받는다. 브랜치 루트의 `ontology/`와 `source/`를 이 저장소의
`data/` 아래에 배치한다.

`data/external/`과 `data/raw/`는 로컬 참고·원본 자료이므로 Hugging Face 릴리스에서
제외한다. QA는 추후 다시 생성해 `data/qa/`에 둔다.
