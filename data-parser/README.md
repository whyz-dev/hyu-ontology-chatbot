# Data parser

2026학년도 2학기 서울캠퍼스 수강신청 자료만 수집하고 구조화한다. PDF의 OCR
단어와 좌표를 원문 근거로 사용한다. Qwen은 좌표로 이미 확정된 열에 영어 의미 키를
붙일 뿐, 제목·헤더·셀 값이나 근거 토큰을 생성하지 않는다.

```bash
uv run python data-parser/crawler
uv run python data-parser/parser --pages 1-75
```

`--pages`는 책자에 인쇄된 페이지 번호이며 쉼표와 범위를 지원한다. 캐시를 무시하고
OCR부터 다시 실행하려면 `--force`를 붙인다. 실행 전 Ollama에 `qwen3.5:9b`가 있어야 한다.
대표 표만 다시 확인할 때는 `--pages 40-41`처럼 범위를 줄일 수 있다.

처리 순서는 다음과 같다.

1. PDF에 포함된 페이지 JPEG를 재인코딩 없이 추출한다.
2. PaddleOCR로 단어, 좌표, 신뢰도를 만들고 공식 e-book 텍스트와 단어 존재 여부를 비교한다.
3. OpenCV 선 검출로 행·열과 `rowspan`/`colspan` 범위를 복원한다.
4. 빈 셀을 포함한 물리 `Cell` 객체와 제목, 다단 헤더 경로, 모든 셀 값과 OCR token ID를 좌표만으로 확정한다.
5. Qwen은 각 확정 열에 고유한 영어 `semantic key`만 부여한다.
6. 문자열·숫자·좌표·병합 범위·OCR 신뢰도·e-book 일치와 문서 전체 용어 일관성을 검사한다.

제목 없는 다음 페이지 표는 직전 페이지와 모든 헤더 경로가 같을 때만 같은 표로 연결한다.
이때 제목 근거와 semantic key를 이전 페이지에서 계승하지만 셀 값은 현재 페이지 OCR로
독립 검증한다.

사용자용 결과는 세 개뿐이다.

- `data/processed/document.jsonl`: block, raw cell, normalized row, semantic annotation과 검증 상태를 모두 포함한 canonical 문서
- `data/processed/review.jsonl`: 자동 승인하지 않은 행만 모은 검수 파일
- `data/processed/preview.md`: 줄글과 표를 읽기 순서로 재구성한 Markdown

OCR, layout, Qwen checkpoint와 실행 manifest는 재실행을 위한 내부 자료이므로
`data/cache/` 아래에 저장한다. 자동 승인 행은 `document.jsonl`에서
`validation_status=auto_accepted`로 필터링한다.

`verified`는 원문 근거와 자동 규칙을 모두 통과했다는 뜻이지, OCR이 현실의 정답임을
보증한다는 뜻은 아니다. 예를 들어 동일 문서에서 더 빈번한 유사 기관명이 발견되면 값을
자동 수정하지 않고 `review.jsonl`로 보낸다.

## 페이지 Markdown 복원

좌표 기반 표 구조 대신 PDF 페이지 전체를 사람이 읽을 수 있는 Markdown으로 복원하려면
별도 복원기를 실행한다.

```bash
uv run python data-parser/parser/restoration.py --pages 1-75
```

복원기는 PDF 이미지와 공식 e-book 텍스트를 Qwen에 함께 제공하고 숫자·단어 일치도를
다시 검사한다. 결과는 `data/restored/`에 저장되며, 사람이 확인해 확정한 데이터만
별도 수동 데이터셋으로 승격한다.
