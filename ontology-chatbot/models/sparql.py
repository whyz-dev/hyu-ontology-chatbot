"""SPARQL 질의 정합, 안전성 검사, 로컬 SELECT 실행을 담당한다.

이 모듈은 ontology graph의 구체적인 색인 구조를 알지 않는다. 알려진
vocabulary 판정과 RDF term 직렬화는 호출자가 주입하므로 ``ontology``
모듈과 순환 참조 없이 재사용할 수 있다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from itertools import islice
from typing import Protocol

from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

MAX_RESULT_ROWS = 50

FORBIDDEN_KEYWORDS = {
    "INSERT",
    "DELETE",
    "LOAD",
    "CLEAR",
    "CREATE",
    "DROP",
    "MOVE",
    "COPY",
    "ADD",
    "SERVICE",
    "FROM",
}
NON_SELECT_QUERY_FORMS = {"ASK", "CONSTRUCT", "DESCRIBE"}


class ResolvedTermLike(Protocol):
    """구체 모델에 의존하지 않고 placeholder를 치환하기 위한 최소 계약."""

    placeholder_iri: str
    selected_iri: str


def _query_tokens(query: str) -> list[tuple[str, int, int, int]]:
    """문자열·주석·IRI를 제외한 SPARQL 영문 token과 중괄호 깊이를 반환한다.

    금지 키워드를 단순 부분 문자열로 찾으면 literal 안의 ``FROM`` 같은 정상 텍스트도
    차단된다. 반대로 주석이나 IRI 내부 문자열은 명령으로 해석하면 안 되므로 작은 lexical
    scanner로 실행 문법에 해당하는 token만 분리한다.
    """

    tokens: list[tuple[str, int, int, int]] = []
    depth = 0
    index = 0
    length = len(query)
    while index < length:
        char = query[index]
        if char == "#":
            newline = query.find("\n", index + 1)
            index = length if newline < 0 else newline + 1
            continue
        if char in {'"', "'"}:
            delimiter = char * 3 if query.startswith(char * 3, index) else char
            index += len(delimiter)
            while index < length:
                if query.startswith(delimiter, index):
                    index += len(delimiter)
                    break
                if query[index] == "\\":
                    index += 2
                else:
                    index += 1
            continue
        if char == "<":
            iri = re.match(r"<[^<>{}\s]*>", query[index:])
            if iri:
                index += len(iri.group(0))
                continue
        if char == "{":
            depth += 1
            index += 1
            continue
        if char == "}":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if char.isascii() and char.isalpha():
            end = index + 1
            while end < length and (
                query[end].isascii() and (query[end].isalnum() or query[end] == "_")
            ):
                end += 1
            previous = query[index - 1] if index else ""
            following = query[end] if end < length else ""
            if previous not in {"?", "$", ":"} and following != ":":
                tokens.append((query[index:end].upper(), depth, index, end))
            index = end
            continue
        index += 1
    return tokens


def _limit_query(
    query: str,
    tokens: list[tuple[str, int, int, int]],
    *,
    max_result_rows: int = MAX_RESULT_ROWS,
) -> str:
    """subquery LIMIT과 무관하게 바깥 SELECT 결과를 최대 행 수로 제한한다.

    중괄호 깊이 0의 LIMIT만 외부 질의 제한으로 인정한다. 제한이 없으면 추가하고 더 큰
    값이면 낮추되, 이미 더 엄격한 사용자 제한은 유지한다.
    """

    outer_limits = [token for token in tokens if token[0] == "LIMIT" and token[1] == 0]
    if not outer_limits:
        return f"{query.rstrip()}\nLIMIT {max_result_rows}"
    _, _, _, token_end = outer_limits[0]
    value_match = re.match(r"\s+(\d+)", query[token_end:])
    if value_match is None:
        raise ValueError("Invalid outer LIMIT clause")
    value_start = token_end + value_match.start(1)
    value_end = token_end + value_match.end(1)
    if int(value_match.group(1)) <= max_result_rows:
        return query
    return query[:value_start] + str(max_result_rows) + query[value_end:]


def validate_query_vocabulary(
    sparql: str,
    *,
    is_known_vocabulary_iri: Callable[[str], bool],
    namespaces: Mapping[str, object],
    query_term_prefix: str,
    xsd_local_names: frozenset[str] | set[str],
) -> None:
    """hallucinated absolute IRI와 vocabulary term을 실행 전에 거부한다.

    질문 resource용 placeholder만 예외로 두고, 절대 IRI와 CURIE 모두 호출자가 제공한
    graph 기반 판정 함수로 검사한다. 따라서 문법상 올바르더라도 ontology에 존재하지
    않는 LLM 생성 property는 실행되지 않는다.
    """

    for iri in re.findall(r"<([^<>\s]+)>", sparql):
        if iri.startswith(query_term_prefix):
            continue
        if not is_known_vocabulary_iri(iri):
            raise ValueError(f"Query uses an unknown absolute IRI: <{iri}>")

    allowed_prefixes = "|".join(re.escape(prefix) for prefix in namespaces)
    curie_pattern = rf"(?<![\w?-])({allowed_prefixes}):([A-Za-z][\w-]*)"
    for prefix, local_name in re.findall(curie_pattern, sparql):
        if prefix == "xsd" and local_name in xsd_local_names:
            continue
        iri = str(namespaces[prefix]) + local_name
        if not is_known_vocabulary_iri(iri):
            raise ValueError(
                f"Query uses an unknown vocabulary term: {prefix}:{local_name}"
            )


def substitute_terms(
    sparql: str,
    terms: Sequence[ResolvedTermLike],
    *,
    query_term_prefix: str,
) -> str:
    """query placeholder를 검증된 ontology resource IRI로 치환한다.

    선언된 placeholder가 query에 없거나 치환 후 하나라도 남으면 실패한다. 불완전한
    정합 결과로 넓은 질의를 실행하는 대신 해당 질문을 명시적으로 중단하기 위한 제한이다.
    """

    refined = sparql
    for term in terms:
        placeholder = f"<{term.placeholder_iri}>"
        if placeholder not in refined:
            raise ValueError(
                f"Query does not contain term placeholder: {term.placeholder_iri}"
            )
        refined = refined.replace(placeholder, f"<{term.selected_iri}>")
    if query_term_prefix in refined:
        raise ValueError("Query still contains unresolved term placeholders")
    return refined


def safe_select_query(
    sparql: str,
    *,
    max_result_rows: int = MAX_RESULT_ROWS,
) -> str:
    """로컬 읽기 전용 SELECT만 허용하고 결과 행 수를 제한한다.

    Markdown fence는 제거하지만 질의를 수정해 복구하지는 않는다. mutation, 원격
    ``SERVICE``와 외부 dataset ``FROM``을 차단한 뒤 RDFLib parser로 전체 문법을 다시
    확인한다.
    """

    query = sparql.strip()
    query = re.sub(r"^```(?:sparql)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    tokens = _query_tokens(query)
    keywords = {token[0] for token in tokens}
    if keywords & FORBIDDEN_KEYWORDS:
        raise ValueError("Only local read-only SPARQL is allowed")
    if "SELECT" not in keywords:
        raise ValueError("Only SELECT queries are allowed")
    if keywords & NON_SELECT_QUERY_FORMS:
        raise ValueError("Only SELECT queries are allowed")
    query = _limit_query(query, tokens, max_result_rows=max_result_rows)

    try:
        parseQuery(query)
    except Exception as error:  # RDFLib parser의 예외 계층이 공개 계약이 아니다.
        raise ValueError(f"Invalid SPARQL: {error}") from error
    return query


def execute_select(
    graph: Graph,
    sparql: str,
    *,
    term_value: Callable[[object], dict[str, object]],
    max_result_rows: int = MAX_RESULT_ROWS,
) -> dict[str, object]:
    """안전성 검사를 통과한 로컬 SELECT를 JSON 호환 결과로 만든다.

    RDF term 직렬화는 ontology 모듈에서 주입받아 이 모듈이 label index를 알 필요가 없다.
    같은 binding이 중복되는 경우 최초 행만 유지하고, 안전 검사와 별개로 iteration도 최대
    행 수에서 멈춘다.
    """

    query = safe_select_query(sparql, max_result_rows=max_result_rows)
    result = graph.query(query)
    variables = [str(variable) for variable in result.vars or []]
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for result_row in islice(result, max_result_rows):
        values = result_row.asdict()
        row = {str(variable): term_value(value) for variable, value in values.items()}
        # 변수 순서가 달라도 같은 binding이면 동일하도록 canonical JSON을 사용한다.
        fingerprint = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if fingerprint not in seen:
            rows.append(row)
            seen.add(fingerprint)
    return {"variables": variables, "rows": rows, "row_count": len(rows)}
