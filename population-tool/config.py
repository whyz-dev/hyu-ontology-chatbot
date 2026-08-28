"""Population 실행 경로, 모델, 데이터 계약 버전을 정의한다."""

from __future__ import annotations

from pathlib import Path

# 실행 위치가 달라도 산출물이 같은 곳에 쌓이도록 저장소 루트를 기준으로 잡는다.
TOOL_ROOT = Path(__file__).resolve().parent
ROOT = TOOL_ROOT.parent
DATA_ROOT = ROOT / "data" / "hyu-ontology"
DATASET_ROOT = ROOT / "data"
CANONICAL_DOCUMENT = DATASET_ROOT / "source" / "document.md"
PROFILE_ROOT = DATA_ROOT / "published" / "profile"
APPLICATION_PROFILE_PATH = PROFILE_ROOT / "application-profile.ttl"
UPSTREAM_LOCK_PATH = PROFILE_ROOT / "upstream.lock.json"

# 모델 태그와 데이터 계약 버전은 manifest에 기록되는 재현성 정보다.
GENERATION_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_URL = "http://127.0.0.1:11434"
PROMPT_VERSION = "population-prompt-v4"
SCHEMA_VERSION = "hyu-manual-markdown-v1"
UNIT_SCHEMA_VERSION = "population-page-v2"
CANDIDATE_SCHEMA_VERSION = "population-candidate-v3"
