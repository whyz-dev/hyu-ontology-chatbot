# Data generator

UVaBot의 실제 대학 사용자 질문에서는 표현 방식만 가져오고, 한양대학교 2026학년도
2학기 학사안내와 공지에서 사실과 근거를 가져와 작은 대화형 질문 데이터셋을 생성한다.

기본 실행은 수강신청 핵심 내용이 있는 인쇄 페이지 2~16과 최신 공지를 사용한다. 모델이 전체
문서에서 근거를 고르게 하지 않고, 생성기에서 서로 다른 표 행·문단을 먼저 evidence unit으로
분리해 질문마다 하나씩 배정한다. 한 근거에서 여러 표현을 만드는 방식은 사용하지 않는다.
기본값은 고유 근거 140개에서 후보를 하나씩 만들고, 질문·Query IR·근거가 모두 겹치지 않는
100개만 고른다. 같은 페이지라도 서로 다른 표 행이나 문장은 별도 근거다. 질문 유형은 내용에
맞게 생성한 뒤 편중만 줄인다. 선별기는 독립 질문 60%와 clarification,
contextual follow-up, user correction, confirmation 대화를 각각 10%로 목표하지만, 자동 검증을
통과한 고유 후보가 부족한 대화 형식은 억지로 채우지 않는다.

```bash
uv run python data-generator
```

10개 단위의 작은 실행으로 시험할 수 있다.

```bash
uv run python data-generator --target 10 --extra-rounds 0
```

기존 실행과 다른 seed·표현 예시를 사용해 추가 후보를 만들 때는 완료한 회차 수를 넘긴다.

```bash
uv run python data-generator --target 30 --extra-rounds 2 --round-offset 18
```

학사안내 전체를 사용하려면 컨텍스트를 함께 늘린다.

```bash
uv run python data-generator --target 100 --pages all --num-ctx 262144
```

결과는 두 파일만 생성한다.

- `data/qa/dataset.jsonl`: 생성된 대화, Query IR, 자동 검증 결과
- `data/qa/preview.md`: 사람이 읽고 검수하기 위한 보기

각 레코드의 `distinctness`에는 질문, Query IR, 근거 묶음 fingerprint와 개별
`evidence_ids`가 기록된다. 100개를 고유하게 채우지 못하면 유사 근거로 자동 보충하지 않고
실행을 실패시킨다. 이 경우 보충 회차를 늘리거나 공식 근거 문서를 추가해야 한다.
`reference_answer`는 반드시 해당 evidence의 연속 문자열이어야 하며, 자동 검증을 통과하지
못한 후보는 최종 데이터에 포함하지 않는다.

두 파일의 모든 레코드는 `needs_human_review` 상태다. UVaBot의 답변과 대학 고유 사실은
사용하지 않는다. Valladolid 관련 표현, 근거에 없는 인용문, 숫자 불일치 등이 발견된 후보는
삭제하지 않고 `automatic_validation.status=failed`와 문제 사유를 남겨 사람이 수정하거나
거절할 수 있게 한다.
