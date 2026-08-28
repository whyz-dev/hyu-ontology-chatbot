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

## 브랜치 규칙

각 패키지는 최신 `main`에서 `feat/<패키지 이름>`으로 분기한다. 기능 검증 후
`main`에 merge하고, 다음 패키지는 갱신된 `main`에서 다시 분기한다. 아직 검증이 끝나지
않은 패키지는 feature branch에만 유지한다.

## 개발

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync
```
