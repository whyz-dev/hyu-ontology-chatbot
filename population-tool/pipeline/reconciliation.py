"""자유 생성된 용어와 entity를 canonical IRI에 정합한다."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from adapters.ollama import OllamaClient, cosine
from adapters.rdf import candidate_entity_iri, candidate_term_iri, expanded_terms
from adapters.sources import application_profile_terms, ontology_term
from adapters.storage import (
    read_json,
    sha256_text,
    stable_json,
    write_json,
    write_jsonl,
)
from config import EMBEDDING_MODEL
from domain.vocabulary import ONTOLOGY_TERMS, RESOURCE_IRI_ROOT, VOCAB_ROOT
from rapidfuzz import fuzz

POSITIVE = {"가능", "허용", "포함", "참여", "인정", "승인", "수 있다"}
NEGATIVE = {"불가", "금지", "제외", "못", "없", "않", "제한", "반려", "수 없다"}
GLOBAL_ENTITY_TYPES = {
    "https://schema.org/Course",
    "http://www.w3.org/ns/org#Organization",
    "http://www.w3.org/ns/org#OrganizationalUnit",
    "urn:hyu-chatbot:vocab:College",
    "urn:hyu-chatbot:vocab:Department",
    "urn:hyu-chatbot:vocab:StudentCategory",
    "urn:hyu-chatbot:vocab:CourseCategory",
    "urn:hyu-chatbot:vocab:AdministrativeAction",
    "urn:hyu-chatbot:vocab:AcademicTerm",
    "urn:hyu-chatbot:vocab:InformationSystem",
    "urn:hyu-chatbot:vocab:Actor",
}


def normalize_label(value: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value).lower())


def _ngrams(value: str, size: int = 3) -> Counter[str]:
    value = normalize_label(value)
    if len(value) <= size:
        return Counter([value]) if value else Counter()
    return Counter(
        value[index : index + size] for index in range(len(value) - size + 1)
    )


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    denominator = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(
        sum(value * value for value in right.values())
    )
    return numerator / denominator if denominator else 0.0


def lexical_similarity(left: str, right: str) -> float:
    # 띄어쓰기/어순/부분 문자열 차이에 강한 세 문자열 점수 중 최댓값을 쓴다.
    return max(
        fuzz.ratio(normalize_label(left), normalize_label(right)) / 100,
        fuzz.token_set_ratio(left, right) / 100,
        _counter_cosine(_ngrams(left), _ngrams(right)),
    )


def _polarity(value: str) -> set[str]:
    result = set()
    if any(token in value for token in POSITIVE):
        result.add("positive")
    if any(token in value for token in NEGATIVE):
        result.add("negative")
    return result


def polarity_compatible(left: dict[str, object], right: dict[str, object]) -> bool:
    left_polarity = _polarity(f"{left.get('label', '')} {left.get('definition', '')}")
    right_polarity = _polarity(
        f"{right.get('label', '')} {right.get('definition', '')}"
    )
    return not (
        left_polarity == {"positive"}
        and right_polarity == {"negative"}
        or left_polarity == {"negative"}
        and right_polarity == {"positive"}
    )


def _value_compatible(left: list[object], right: list[object]) -> bool:
    if not left and not right:
        return True
    if not left or not right:
        return False
    left_values = {normalize_label(item) for item in left}
    right_values = {normalize_label(item) for item in right}
    return bool(left_values & right_values)


def structure_compatible(left: dict[str, object], right: dict[str, object]) -> bool:
    # 이름이 비슷해도 class/property 종류나 domain/range 의미가 다르면 합치지 않는다.
    if left.get("kind") != right.get("kind"):
        return False
    if left.get("kind") == "class":
        return polarity_compatible(left, right)
    return (
        _value_compatible(list(left.get("domains", [])), list(right.get("domains", [])))
        and _value_compatible(
            list(left.get("ranges", [])), list(right.get("ranges", []))
        )
        and polarity_compatible(left, right)
    )


def embedding_text(term: dict[str, object]) -> str:
    return (
        "한국어 ontology 용어 정합. "
        f"종류: {term.get('kind')}. 이름: {term.get('label')}. "
        f"정의: {term.get('definition')}. "
        f"domain: {', '.join(str(item) for item in term.get('domains', []))}. "
        f"range: {', '.join(str(item) for item in term.get('ranges', []))}."
    )


def _canonical_local_iri(term: dict[str, object]) -> str:
    label = normalize_label(term.get("label"))[:40] or "term"
    digest = sha256_text(stable_json([term.get("kind"), label]))[:12]
    return f"{VOCAB_ROOT}{label}-{digest}"


def _upstream_terms(graph: object) -> list[dict[str, object]]:
    seen = set()
    result = []
    for iri in ONTOLOGY_TERMS:
        if iri in seen:
            continue
        seen.add(iri)
        result.append(ontology_term(graph, iri))
    for term in application_profile_terms():
        iri = str(term["iri"])
        if iri not in seen:
            seen.add(iri)
            result.append(dict(term))
    return result


def _local_terms(
    candidates: list[dict[str, object]], run_id: str
) -> list[dict[str, object]]:
    result = []
    for candidate in candidates:
        unit_id = str(candidate["unit_id"])
        for term in expanded_terms(candidate):
            if term.get("iri"):
                continue
            value = dict(term)
            value["unit_id"] = unit_id
            value["candidate_iri"] = str(
                candidate_term_iri(run_id, unit_id, str(term["term_id"]))
            )
            value["term_key"] = f"{unit_id}::{term['term_id']}"
            result.append(value)
    return sorted(result, key=lambda item: str(item["term_key"]))


def _embedding_map(
    client: OllamaClient,
    terms: list[dict[str, object]],
    cache_path: Path,
) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        raw = read_json(cache_path)
        if isinstance(raw, dict) and raw.get("model") == EMBEDDING_MODEL:
            cache = {
                str(key): [float(value) for value in vector]
                for key, vector in raw.get("vectors", {}).items()
            }
    # 용어 설명의 hash를 key로 사용해 동일 embedding을 재호출하지 않는다.
    texts = {sha256_text(embedding_text(term)): embedding_text(term) for term in terms}
    missing = [(key, text) for key, text in texts.items() if key not in cache]
    for start in range(0, len(missing), 64):
        batch = missing[start : start + 64]
        vectors = client.embed(EMBEDDING_MODEL, [text for _, text in batch])
        cache.update({key: vector for (key, _), vector in zip(batch, vectors)})
        write_json(cache_path, {"model": EMBEDDING_MODEL, "vectors": cache})
    return cache


def _compare(
    left: dict[str, object],
    right: dict[str, object],
    vectors: dict[str, list[float]],
) -> dict[str, object]:
    lexical = lexical_similarity(
        str(left.get("label", "")), str(right.get("label", ""))
    )
    left_vector = vectors[sha256_text(embedding_text(left))]
    right_vector = vectors[sha256_text(embedding_text(right))]
    semantic = cosine(left_vector, right_vector)
    structure = structure_compatible(left, right)
    polarity = polarity_compatible(left, right)
    # 문자열과 의미 점수를 결합하되 자동 병합 여부는 구조·극성 조건도 따로 본다.
    return {
        "lexical_score": round(lexical, 6),
        "embedding_score": round(semantic, 6),
        "combined_score": round(0.45 * lexical + 0.55 * semantic, 6),
        "structure_compatible": structure,
        "polarity_compatible": polarity,
    }


def _is_auto(score: dict[str, object]) -> bool:
    # 자동 병합은 보수적으로 두고 애매한 후보는 review 단계로 넘긴다.
    return (
        score["combined_score"] >= 0.93
        and score["lexical_score"] >= 0.85
        and score["embedding_score"] >= 0.90
        and score["structure_compatible"]
        and score["polarity_compatible"]
    )


def _needs_review(score: dict[str, object]) -> bool:
    return score["combined_score"] >= 0.78 or score["embedding_score"] >= 0.82


def reconcile_terms(
    run_id: str,
    candidates: list[dict[str, object]],
    graph: object,
    client: OllamaClient,
    run_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    upstream = _upstream_terms(graph)
    local = _local_terms(candidates, run_id)
    # 현재 profile은 새 schema 용어 생성을 허용하지 않으므로 대부분 local이 비어
    # 있다. 이때는 embedding 모델이나 불필요한 upstream embedding을 요구하지 않는다.
    vectors = (
        _embedding_map(client, [*upstream, *local], run_root / "embeddings.json")
        if local
        else {}
    )
    alignments: list[dict[str, object]] = []
    # 앞에서 distinct로 확정된 로컬 용어도 뒤 용어의 정합 후보가 된다.
    representatives: list[dict[str, object]] = []
    for term in local:
        comparisons: list[tuple[dict[str, object], dict[str, object], str]] = []
        for target in [*upstream, *representatives]:
            if target.get("kind") != term.get("kind"):
                continue
            target_iri = str(target.get("iri") or target.get("canonical_iri"))
            comparisons.append((_compare(term, target, vectors), target, target_iri))
        comparisons.sort(
            key=lambda item: (
                item[0]["combined_score"],
                item[0]["lexical_score"],
                item[2],
            ),
            reverse=True,
        )
        best = comparisons[0] if comparisons else None
        own_iri = _canonical_local_iri(term)
        if (
            best
            and normalize_label(term["label"]) == normalize_label(best[1]["label"])
            and structure_compatible(term, best[1])
        ):
            status = "auto_merged"
            target_iri = best[2]
            reason = "exact normalized label and compatible structure"
            score = best[0]
        elif best and _is_auto(best[0]):
            status = "auto_merged"
            target_iri = best[2]
            reason = "high lexical and embedding score with compatible structure"
            score = best[0]
        elif best and _needs_review(best[0]):
            status = "needs_review"
            target_iri = best[2]
            reason = "similarity candidate requires human decision"
            score = best[0]
        else:
            status = "distinct"
            target_iri = own_iri
            reason = "no sufficiently similar compatible term"
            score = best[0] if best else None
        alignment_id = sha256_text(
            stable_json([term["candidate_iri"], target_iri, term.get("kind")])
        )[:24]
        record = {
            "schema_version": "term-alignment-v1",
            "alignment_id": alignment_id,
            "term_key": term["term_key"],
            "unit_id": term["unit_id"],
            "candidate_iri": term["candidate_iri"],
            "candidate": {
                key: term.get(key)
                for key in (
                    "term_id",
                    "kind",
                    "label",
                    "definition",
                    "domains",
                    "ranges",
                    "implicit",
                )
            },
            "status": status,
            "target_iri": target_iri,
            "distinct_iri": own_iri,
            "target": (
                {
                    key: best[1].get(key)
                    for key in (
                        "iri",
                        "kind",
                        "label",
                        "definition",
                        "domains",
                        "ranges",
                    )
                }
                if best
                else None
            ),
            "score": score,
            "reason": reason,
        }
        alignments.append(record)
        if status == "distinct":
            representative = dict(term)
            representative["canonical_iri"] = own_iri
            representatives.append(representative)
    write_jsonl(run_root / "alignments.jsonl", alignments)
    entity_mappings = reconcile_entities(run_id, candidates)
    write_jsonl(run_root / "entity-mappings.jsonl", entity_mappings)
    return alignments, entity_mappings


def reconcile_entities(
    run_id: str, candidates: list[dict[str, object]]
) -> list[dict[str, object]]:
    # 개체는 embedding으로 합치지 않는다. 규칙·기간·절차처럼 문맥 의존적인
    # instance는 이름이 같아도 페이지 scope를 유지하고, 공식 식별자 또는 안정된
    # 분류/조직 개체만 정확한 이름과 type으로 통합한다.
    canonical: dict[tuple[str, str, tuple[str, ...], str], str] = {}
    result = []
    for candidate in sorted(candidates, key=lambda item: str(item["unit_id"])):
        unit_id = str(candidate["unit_id"])
        entity_locators: dict[str, set[str]] = {}
        for fact in candidate.get("facts", []):
            if not isinstance(fact, dict):
                continue
            locator = str(fact.get("evidence_locator", ""))
            entity_locators.setdefault(str(fact.get("subject", "")), set()).add(locator)
            obj = fact.get("object")
            if isinstance(obj, dict) and obj.get("kind") == "entity":
                entity_locators.setdefault(str(obj.get("value", "")), set()).add(
                    locator
                )
        for entity in candidate.get("entities", []):
            if not isinstance(entity, dict):
                continue
            identifier = normalize_label(entity.get("identifier") or "")
            label = normalize_label(entity.get("label"))
            raw_types = tuple(sorted(str(item) for item in entity.get("types", [])))
            types = tuple(normalize_label(item) for item in raw_types)
            globally_scoped = (
                bool(identifier)
                or bool(raw_types)
                and all(item in GLOBAL_ENTITY_TYPES for item in raw_types)
            )
            locator_scope = sorted(
                entity_locators.get(str(entity.get("entity_id")), set())
            )
            scope = (
                "global"
                if globally_scoped
                else f"{unit_id}:{sha256_text(stable_json(locator_scope))[:12]}"
            )
            key = (identifier, label, types, scope)
            if key not in canonical:
                slug = label[:40] or "entity"
                digest = sha256_text(stable_json(key))[:12]
                canonical[key] = f"{RESOURCE_IRI_ROOT}{slug}-{digest}"
            entity_id = str(entity["entity_id"])
            result.append(
                {
                    "schema_version": "entity-mapping-v1",
                    "entity_key": f"{unit_id}::{entity_id}",
                    "candidate_iri": str(
                        candidate_entity_iri(run_id, unit_id, entity_id)
                    ),
                    "canonical_iri": canonical[key],
                    "label": entity.get("label"),
                    "types": entity.get("types", []),
                    "method": (
                        "exact identifier/label/type"
                        if globally_scoped
                        else "exact label/type within page scope"
                    ),
                }
            )
    return result
