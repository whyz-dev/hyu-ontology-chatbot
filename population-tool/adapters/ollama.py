"""Ollama의 구조화 생성과 embedding API 호출을 감싼다."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error
        if not isinstance(value, dict):
            raise TypeError("Ollama returned a non-object response")
        return value

    def models(self) -> dict[str, str]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=10
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Cannot connect to Ollama: {error}") from error
        return {
            str(item.get("name")): str(item.get("digest", ""))
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        }

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        retries: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        # 샘플링을 고정하고 JSON Schema를 Ollama에 전달해 출력 형태를 제한한다.
        # 같은 seed의 길이 초과 응답을 반복해도 결과가 달라지지 않으므로 기본 재시도는
        # 하지 않는다. 호출자는 정말 필요한 경우에만 명시적으로 횟수를 늘릴 수 있다.
        for attempt in range(retries + 1):
            try:
                response = self._post(
                    "/api/generate",
                    {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        # Qwen 3.5가 JSON을 thinking 필드에만 쓰지 않도록 한다.
                        "think": False,
                        "format": schema,
                        "options": {
                            "temperature": 0,
                            "seed": 42,
                            "num_ctx": 32768,
                            # 의미 단위 하나가 이보다 긴 JSON을 요구하면 정상 추출이 아니다.
                            # 상한 없이 malformed 응답을 계속 생성하는 경우도 차단한다.
                            "num_predict": 8192,
                        },
                    },
                )
                if response.get("done_reason") == "length":
                    raise ValueError("Ollama structured response exceeded token limit")
                raw_response = response.get("response")
                if not isinstance(raw_response, str) or not raw_response.strip():
                    raise ValueError("Ollama returned an empty structured response")
                value = json.loads(raw_response)
                if not isinstance(value, dict):
                    raise TypeError("structured response is not an object")
                metadata = {
                    "model": model,
                    "attempt": attempt + 1,
                    "total_duration": response.get("total_duration"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                }
                return value, metadata
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        raise RuntimeError(
            f"Ollama structured output failed after {retries + 1} attempts: {last_error}"
        )

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        response = self._post("/api/embed", {"model": model, "input": inputs})
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise RuntimeError("Ollama returned an invalid embedding batch")
        return [[float(value) for value in vector] for vector in embeddings]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator if denominator else 0.0
