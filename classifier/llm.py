import json
import logging
import re
import threading
import time

from typing import Any

import requests

from config import (
    ENDPOINTS,
    LLM_API_KEY,
    MAX_RETRIES,
    MAX_TOKENS,
    RETRY_DELAYS,
    TEMPERATURE,
    TIMEOUT,
    TOP_P,
)

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = MAX_TOKENS,
        api_key: str | None = None,
    ):
        if provider not in ENDPOINTS:
            raise ValueError(
                f"Неизвестный provider: {provider}"
            )

        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.endpoint = ENDPOINTS[provider]
        self.api_key = api_key or LLM_API_KEY

        if not self.api_key:
            raise ValueError(
                "Не задан LLM_API_KEY."
            )

        self.local = threading.local()

    def _get_session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()

        return self.local.session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_json(content: str) -> list[dict[str, Any]]:
        content = content.strip()

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("[")
            end = content.rfind("]")

            if start == -1 or end <= start:
                raise LLMError(
                    f"Некорректный JSON: {content[:1000]}"
                )

            parsed = json.loads(content[start:end + 1])

        if not isinstance(parsed, list):
            raise LLMError(
                "Ответ модели должен быть JSON-массивом."
            )

        required_fields = {
            "name",
            "answer",
            "comment",
            "mention",
        }

        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMError(
                    f"Элемент {index} не является объектом."
                )

            missing = required_fields - item.keys()

            if missing:
                raise LLMError(
                    f"В элементе {index} отсутствуют поля: {missing}"
                )

            item["name"] = str(item["name"])
            item["answer"] = str(item["answer"]).strip()
            item["comment"] = str(item["comment"])
            item["mention"] = str(item["mention"])

            if item["answer"] not in {"0", "1"}:
                raise LLMError(
                    f"Некорректный answer: {item['answer']}"
                )

            if item["answer"] == "0":
                item["mention"] = ""

        return parsed

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get_session().post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=TIMEOUT,
                )

                response.raise_for_status()

                body = response.json()

                choice = body["choices"][0]
                content = choice["message"]["content"]

                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )

                if not isinstance(content, str):
                    raise LLMError(
                        "Некорректный формат message.content."
                    )

                predictions = self._extract_json(content)

                return predictions, {
                    "request_id": body.get("id"),
                    "usage": body.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "model": body.get("model", self.model_name),
                }

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "LLM error, attempt %s/%s: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[
                        min(attempt, len(RETRY_DELAYS) - 1)
                    ]
                    time.sleep(delay)

        raise LLMError(
            f"LLM request failed: {last_error}"
        ) from last_error