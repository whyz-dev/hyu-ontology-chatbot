"""`config.ini`를 검증해 불변 실행 설정으로 제공한다.

설정 누락을 코드 기본값으로 조용히 보완하지 않는다. 파일 계약이 달라졌거나 값이
잘못된 경우 애플리케이션 시작 단계에서 즉시 실패하게 하여, 서로 다른 설정으로 실행된
결과가 같은 것처럼 보이는 일을 막는다.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CHATBOT_ROOT / "config.ini"


@dataclass(frozen=True)
class Settings:
    """온톨로지와 LLM 실행에 필요한 검증 완료 설정 묶음.

    Controller는 전역 상수를 흩어 사용하지 않고 이 객체를 통해 동일한 seed와 출력
    한도를 공유한다. frozen dataclass이므로 실행 도중 값이 바뀌지 않는다.
    """

    ontology_path: Path
    model: str
    ollama_url: str
    seed: int
    query_max_tokens: int
    answer_max_tokens: int


def _require_names(actual: set[str], expected: set[str], *, location: str) -> None:
    """INI section 또는 key 집합이 선언된 계약과 정확히 같은지 검사한다."""

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Invalid config keys at {location}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _text(value: str, *, location: str) -> str:
    """문자열 설정의 주변 공백을 제거하고 빈 값을 거부한다."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Expected a non-empty string at {location}")
    return normalized


def _positive_integer(value: str, *, location: str) -> int:
    """token 한도와 재현 seed에 사용할 양의 정수만 허용한다."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"Expected an integer at {location}") from error
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer at {location}")
    return parsed


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """명시된 INI를 단 한 번 읽어 검증된 :class:`Settings`로 변환한다.

    section, DEFAULT 값, 각 section의 key를 모두 검사한다. 상대 ontology 경로는
    프로세스의 현재 작업 디렉터리가 아니라 설정 파일 위치를 기준으로 해석한다.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Chatbot config is missing: {path}")
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except configparser.Error as error:
        raise ValueError(f"Malformed chatbot config: {path}: {error}") from error
    # 허용 목록 방식으로 검사해 오타 난 설정이 무시된 채 실행되는 것을 막는다.
    _require_names(
        set(parser.sections()),
        {"ontology", "ollama", "generation"},
        location="root",
    )
    _require_names(set(parser.defaults()), set(), location="DEFAULT")
    expected_options = {
        "ontology": {"path"},
        "ollama": {"model", "url", "seed"},
        "generation": {"query_max_tokens", "answer_max_tokens"},
    }
    for section, expected in expected_options.items():
        _require_names(set(parser.options(section)), expected, location=section)

    ontology_path = Path(_text(parser["ontology"]["path"], location="ontology.path"))
    if not ontology_path.is_absolute():
        ontology_path = (path.parent / ontology_path).resolve()
    return Settings(
        ontology_path=ontology_path,
        model=_text(parser["ollama"]["model"], location="ollama.model"),
        ollama_url=_text(parser["ollama"]["url"], location="ollama.url"),
        seed=_positive_integer(parser["ollama"]["seed"], location="ollama.seed"),
        query_max_tokens=_positive_integer(
            parser["generation"]["query_max_tokens"],
            location="generation.query_max_tokens",
        ),
        answer_max_tokens=_positive_integer(
            parser["generation"]["answer_max_tokens"],
            location="generation.answer_max_tokens",
        ),
    )


# import 시 검증을 끝내므로 View나 Controller가 별도의 fallback을 만들 필요가 없다.
SETTINGS = load_settings()
