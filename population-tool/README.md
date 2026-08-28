# Population tool

검수된 `data/source/document.md`의 본문과 표를
79개 PDF 페이지 단위로 묶은 뒤, Qwen이
단일 HYU application profile에 맞춘 ontology 사실을 페이지별 Turtle로 저장한다. LLM과 validator
사이에서는 JSON Schema를 안전한 전송 형식으로 사용하지만, 첫 few-shot Turtle의
대응 JSON은 코드가 매번 결정적으로 투영한다. 사람이 별도 JSON 예제를 관리하지 않으며,
few-shot 정답 원본과 최종 산출물은 모두 RDF/OWL Turtle이다. 현재는 고정 application profile 밖의 새 schema
용어를 생성하지 않으며, 검수 완료 후 승인된 사실만 하나의 온톨로지로 발행한다.

PDF를 다시 OCR하지 않으며 `data-parser`나 `data-generator`의 Python 모듈을 import하지
않는다.

`data/`는 Git에 포함하지 않는다. 따라서 실행 전 로컬 수동 데이터셋과
`data/hyu-ontology/published/profile/`을 준비해야 한다.

## 코드 구조

```text
population-tool/
├── __main__.py       # `python population-tool` 진입점
├── cli.py            # 명령과 인자 처리
├── config.py         # 경로, 모델, 데이터 계약 버전
├── domain/           # 핵심 데이터 모델과 IRI/vocabulary 정책
├── pipeline/         # 분할, 추출, 정합, 검증, 발행 workflow
├── review/           # 확정 예제, 용어 검수, 검증·발행 GUI
└── adapters/         # Ollama, RDFLib, 외부 ontology, 파일 저장
```

`domain`은 외부 기술에 의존하지 않는다. `pipeline`은 처리 순서를 조정하고 실제 HTTP,
RDF, 파일 입출력은 `adapters`에 맡긴다. `review`는 확정 few-shot 예제와 통합 Streamlit
GUI를 포함한다. `__main__.py`는 `cli.py`로 실행을 전달하기만 한다.

## Application profile

통일 vocabulary는
`data/hyu-ontology/published/profile/application-profile.ttl`에 고정한다.
프롬프트는 선택된 few-shot이 실제 사용하는 class/property와 다음 모델링 규칙만
매 호출에 제공한다.
ORG, OWL-Time, PROV-O, DCTERMS, SKOS, Schema.org는 선택 가능한 모듈이 아니라 이
profile이 재사용하는 고정 vocabulary다. 따라서 별도 모듈 선택 없이 Qwen을 한 번만
호출해 instance와 fact를 추출한다.

- `schema:Course`: 교과목 자체
- `schema:CourseInstance`: 학기·분반별 개설 강좌
- `org:OrganizationalUnit`: 단과대학·학부·학과·전공 조직
- `hyu:StudentCategory`, `hyu:CourseCategory`: SKOS 기반 학생·교과목 분류
- `time:Interval`/`time:Instant`: 수강신청 기간과 날짜
- `hyu:AcademicRule`: 허용·금지·수치 제한과 그 조건·예외

조건 없는 식별 정보와 조직 계층·절차 순서는 직접 triple로 저장한다. 적용 대상,
기간, 조건 또는 예외가 붙는 허용·금지·제한은 rule instance로 저장한다. canonical
property의 `rdfs:domain/range`는 profile에서 한 번만 선언한다. 현재 검증 범위는 RDF
문법, locator별 정확한 원문 근거, 숫자·날짜·서울 시각, domain/range, 시간 범위의
시작·종료 완전성, provenance 및 미정 candidate IRI 검사다.

## 모델 준비

```bash
uv sync
ollama pull qwen3.5:9b
```

모델은 자동 다운로드하지 않는다. 외부 ontology는 `generate`가 lock 파일의 checksum을
검사해 없을 때만 다운로드한다.

- 생성: `qwen3.5:9b`
- 개체 정합: 공식 식별자 또는 정확한 이름·type·scope 일치

## 실행 순서

```bash
uv run python population-tool generate --run-id 2026-2-md-v1
uv run python population-tool review --run-id 2026-2-md-v1
```

중단된 같은 실행만 `generate --run-id 2026-2-md-v1 --resume`으로 이어간다.

`generate`는 외부 ontology 확보, authoritative `document.md`/manifest 해시 검증,
79개 페이지 구성,
확정 few-shot 로딩, Qwen 추출과 페이지별 draft TTL 생성을 한 번에 수행한다. `review`는
페이지 후보 검수, 용어·개체 통합, RDF·근거 검증과 최종 발행을 한 Streamlit 화면에서
수행한다. 먼저 `생성 결과` 탭에서 페이지 원문과 Turtle을 나란히 보고 각 페이지를
승인·거절하거나 구조화 JSON을 수정 승인한다. 79개 페이지의 현재 후보가 모두 승인되어야
통합과 발행을 진행할 수 있다. 예제 원본은
`data/hyu-ontology/published/profile/examples/*.ttl`이며 별도 승인 상태를 두지 않는다.

현재 canonical 문서는 79개 page evidence로 구성된다. 전체 실행 전
소수 페이지만 모델 출력 형태를 확인하려면 아래처럼 실행할 수 있다. `--limit` 실행은 전체
run 검증이나 최종 발행용이 아니다.

```bash
uv run python population-tool generate --run-id 2026-2-smoke --limit 5 --resume
```

## 산출물

```text
data/hyu-ontology/
├── upstream/                         # checksum 고정 외부 ontology
├── runs/<run-id>/
│   ├── candidates.jsonl              # resume용 한 줄 checkpoint
│   ├── population-decisions.jsonl     # 페이지 승인·거절·수정 승인 이력
│   └── ...                            # unit, 정합, 검증, manifest
├── draft/<run-id>/<document-id>/    # fact가 있는 page-XXX.ttl
└── published/
    ├── profile/
    │   ├── application-profile.ttl
    │   ├── upstream.lock.json
    │   ├── fewshot_examples.json
    │   └── examples/*.ttl
    └── hyu-course-registration-2026-2.ttl
```

표는 각 행에 header와 section 문맥을 붙이며 헤더 행 자체도 locator로 보존한다. 페이지를
넘긴 빈 헤더 표는 직전 페이지의 마지막 표와 열 수·section이 모두 맞을 때만 열 문맥을 잇는다.
공통 provenance schema는 profile에 한 번만 선언하고, 페이지 TTL에는 실제 domain fact와
그 fact의 block/row locator만 기록한다. 최종 TTL에는 canonical instance triple과 각 triple의
`rdf:Statement` assertion이 함께 들어간다. 미검수 페이지, 부분 생성, 미결 alignment,
candidate IRI, RDF 또는 근거 검증 오류가 하나라도 있으면 GUI의 발행 버튼은 활성화되지 않는다.

`candidates.jsonl`은 중단 후 재개를 위한 내부 파일이므로 직접 읽지 않는다. 사람이
검수하는 주 산출물은 `draft/`의 페이지 Turtle이며 GUI가 원문과 함께 보여준다. 부분
생성도 유효하게 남은 fact를 draft TTL로 보여주지만 승인할 수 없고, GUI에서 누락·오류를
수정해 다시 검증해야 한다. 확실한 domain fact가 없는 페이지만 TTL 없이 `no_fact`로 남긴다.

## 코드 검증

```bash
uv run --with ruff ruff check population-tool
uv run --with ruff ruff format --check population-tool
uv run python population-tool --help
```

실제 Ollama 추출 smoke test는 `generate --limit`으로 수행한다.
