import argparse
import csv
import json
import logging
import os
import re
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import urllib3

from crowd_sdk.tagme import TagmeClientAdvanced


# ============================================================
# CONFIGURATION
# ============================================================

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid

import requests
import urllib3

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from crowd_sdk.tagme import TagmeClientAdvanced


# ============================================================
# CONFIGURATION
# ============================================================

ENDPOINTS = {
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",
    "aihf": "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
    "sigma": "https://tagme.sigma.sbrf.ru/ai/llm/v1/chat/completions",
}

CFG = {
    "direct": "~/configs/.ml_config.cfg",
    "aihf": "~/configs/.ml_config.cfg",
    "sigma": "~/configs/.ml_config_sigma.cfg",
}

DEFAULT_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 1.0

MAX_TOKENS = 8000
TIMEOUT = (60, 180)

# Количество попыток обращения к LLM.
MAX_RETRIES = 3
RETRY_DELAY = 2

INPUT_FIELD = "part_system_prompt"
CHECKPOINT_EVERY = 10

IGNORE_NAMES = {"fork"}

RUNS_DIR = Path("runs")
REGISTRY_PATH = RUNS_DIR / "runs_registry.csv"

CSV_FIELDS = [
    "timestamp",
    "model",
    "id",
    "accuracy",
]

JSON_FIELDS = [
    "id",
    "model",
    "timestamp",
    "gt_ground_truth",
    "prediction",
]

REGISTRY_FIELDS = [
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "model",
    "provider",
    "temperature",
    "top_p",
    "max_tokens",
    "workers",
    "max_retries",
    "dataset_path",
    "dataset_hash",
    "system_prompt_file",
    "system_prompt_hash",
    "source_files_map",
    "source_files_map_hash",
    "total_records",
    "processed_records",
    "evaluated_records",
    "correct_records",
    "failed_records",
    "invalid_gold_records",
    "coverage",
    "processing_coverage",
    "exact_match_accuracy",
    "error",
    "results_path",
    "metrics_path",
]

_thread_local = threading.local()
_registry_lock = threading.Lock()

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# RUN TRACKING
# ============================================================

def get_timestamp() -> str:
    """Возвращает локальное время с часовым поясом."""

    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def get_file_hash(path: Path) -> str:
    """Вычисляет SHA-256 файла."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_text_hash(value: str) -> str:
    """Вычисляет SHA-256 текстовой строки."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def create_run_id() -> str:
    """Создаёт уникальный идентификатор запуска."""

    return (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )


def save_run_metadata(
    metadata: dict[str, Any],
    metadata_path: Path,
):
    """Атомарно сохраняет метаданные запуска."""

    atomic_save_json(
        metadata,
        metadata_path,
    )


def save_registry_entry(
    entry: dict[str, Any],
):
    """
    Создаёт или обновляет запись в общем реестре запусков.

    Один run_id соответствует одной строке.
    """

    with _registry_lock:

        REGISTRY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = []

        if REGISTRY_PATH.exists():

            with REGISTRY_PATH.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                existing = list(
                    csv.DictReader(file)
                )

        existing = [
            row
            for row in existing
            if row.get("run_id") != entry["run_id"]
        ]

        existing.append({
            field: entry.get(field)
            for field in REGISTRY_FIELDS
        })

        temp_path = REGISTRY_PATH.with_name(
            REGISTRY_PATH.stem
            + ".tmp"
            + REGISTRY_PATH.suffix
        )

        try:

            with temp_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=REGISTRY_FIELDS,
                    extrasaction="ignore",
                )

                writer.writeheader()
                writer.writerows(existing)

            temp_path.replace(REGISTRY_PATH)

        finally:

            if temp_path.exists():
                temp_path.unlink()


def update_run_tracking(
    metadata: dict[str, Any],
    metadata_path: Path,
    registry_entry: dict[str, Any],
    updates: dict[str, Any],
):
    """Обновляет метаданные и общий реестр запусков."""

    metadata.update(updates)
    registry_entry.update(updates)

    save_run_metadata(
        metadata,
        metadata_path,
    )

    save_registry_entry(
        registry_entry,
    )


def get_tracking_metrics(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Выделяет метрики, которые нужно записать в реестр."""

    return {
        "total_records": metrics["total_records"],
        "processed_records": metrics["processed_records"],
        "evaluated_records": metrics["evaluated_records"],
        "correct_records": metrics["correct_records"],
        "failed_records": metrics["failed_records"],
        "invalid_gold_records": metrics["invalid_gold_records"],
        "coverage": metrics["coverage"],
        "processing_coverage": metrics["processing_coverage"],
        "exact_match_accuracy": metrics["exact_match_accuracy"],
    }


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name: str) -> str:
    """Нормализует пробелы и переносы строк."""

    return " ".join(name.split())


_NORMALIZED_IGNORE_NAMES = {
    normalize_name(name).casefold()
    for name in IGNORE_NAMES
}


def normalize_and_filter_name(
    name: Any,
    context: str,
) -> str | None:
    """Нормализует название критерия и исключает служебные имена."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{context}: некорректное поле name."
        )

    normalized = normalize_name(name)

    if normalized.casefold() in _NORMALIZED_IGNORE_NAMES:
        return None

    return normalized


# ============================================================
# ANSWER NORMALIZATION
# ============================================================

def normalize_answer(answer: Any) -> str:
    """Приводит answer к строке '1' или '0'."""

    if isinstance(answer, bool):
        return "1" if answer else "0"

    if isinstance(answer, (int, float)):

        if answer == 1:
            return "1"

        if answer == 0:
            return "0"

    if isinstance(answer, str):

        value = answer.strip()

        if value in {"1", "1.0"}:
            return "1"

        if value in {"0", "0.0"}:
            return "0"

    raise ValueError(
        f"Некорректное значение answer: {answer!r}. "
        "Ожидалось 1 или 0."
    )


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_ground_truth_names(
    items: Any,
    field_description: str,
) -> set[str]:
    """Извлекает положительные категории Ground Truth."""

    names = set()

    for index, item in enumerate(items):

        if not isinstance(item, dict):
            raise ValueError(
                f"{field_description}: элемент {index} "
                "должен быть объектом."
            )

        try:
            answer = normalize_answer(
                item.get("answer")
            )

        except ValueError as exc:

            raise ValueError(
                f"{field_description}, элемент {index}: {exc}"
            ) from exc

        if answer != "1":
            continue

        name = normalize_and_filter_name(
            item.get("name"),
            f"{field_description}, элемент {index}",
        )

        if name is not None:
            names.add(name)

    return names


def extract_predicted_names(
    items: Any,
    field_description: str,
) -> set[str]:
    """Извлекает положительные категории Prediction."""

    names = set()

    for index, item in enumerate(items):

        if not isinstance(item, dict):
            raise ValueError(
                f"{field_description}: элемент {index} "
                "должен быть объектом."
            )

        try:
            answer = normalize_answer(
                item.get("answer")
            )

        except ValueError as exc:

            raise ValueError(
                f"{field_description}, элемент {index}: {exc}"
            ) from exc

        if answer != "1":
            continue

        name = normalize_and_filter_name(
            item.get("name"),
            f"{field_description}, элемент {index}",
        )

        if name is not None:
            names.add(name)

    return names


# ============================================================
# EXPECTED CRITERIA EXTRACTION
# ============================================================

def extract_expected_names(
    user_prompt: str,
    item_id: str,
) -> set[str]:
    """Извлекает полный список критериев из part_system_prompt."""

    try:
        payload = json.loads(user_prompt)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"ID {item_id}: part_system_prompt "
            f"не является валидным JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"ID {item_id}: part_system_prompt "
            "должен содержать JSON-объект."
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            f"ID {item_id}: отсутствует объект data."
        )

    dialogue = data.get("dialogue")

    if not isinstance(dialogue, str):
        raise ValueError(
            f"ID {item_id}: data.dialogue должен быть строкой."
        )

    premarkup = payload.get("premarkup")

    if not isinstance(premarkup, dict):
        raise ValueError(
            f"ID {item_id}: отсутствует объект premarkup."
        )

    classes = premarkup.get("classes")

    if not isinstance(classes, list):
        raise ValueError(
            f"ID {item_id}: premarkup.classes должен быть списком."
        )

    expected_names = set()

    for index, item in enumerate(classes):

        if not isinstance(item, dict):
            raise ValueError(
                f"ID {item_id}: premarkup.classes[{index}] "
                "должен быть объектом."
            )

        name = normalize_and_filter_name(
            item.get("name"),
            f"ID {item_id}, premarkup.classes[{index}]",
        )

        if name is None:
            continue

        if name in expected_names:
            raise ValueError(
                f"ID {item_id}: повторяющийся критерий "
                f"в premarkup.classes: {name!r}."
            )

        expected_names.add(name)

    return expected_names


# ============================================================
# SOURCE FILE MAPPING
# ============================================================

def normalize_source_files(
    value: Any,
    context: str,
) -> list[str]:
    """Проверяет и нормализует список исходных файлов."""

    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        raise ValueError(
            f"{context}: ожидалась строка или список файлов."
        )

    result = []

    for file_path in value:

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(
                f"{context}: обнаружено некорректное имя файла."
            )

        file_path = file_path.strip()

        if file_path not in result:
            result.append(file_path)

    return result


def load_source_files_map(
    mapping_path: Path | None,
) -> dict[str, list[str]]:
    """
    Загружает связи id -> source_files.

    Ожидаемый формат:

    {
        "12345": ["part_01.json", "part_03.json"],
        "12346": ["part_02.json"]
    }
    """

    if mapping_path is None:
        return {}

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Файл связей исходных файлов не найден: {mapping_path}"
        )

    with mapping_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Файл source-files-map должен содержать JSON-объект "
            "с ключами id и списками файлов."
        )

    result = {}

    for item_id, files in data.items():

        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                "В source-files-map обнаружен некорректный id."
            )

        result[item_id] = normalize_source_files(
            files,
            f"source-files-map, id={item_id}",
        )

    return result


def get_source_files(
    original_item: dict[str, Any],
    source_files_map: dict[str, list[str]],
) -> list[str]:
    """
    Объединяет связи из prepared JSON и внешнего mapping.

    Дубликаты удаляются с сохранением порядка.
    """

    item_id = original_item["id"]

    item_files = normalize_source_files(
        original_item.get("source_files"),
        f"prepared JSON, id={item_id}",
    )

    mapped_files = source_files_map.get(
        item_id,
        [],
    )

    result = []

    for file_path in item_files + mapped_files:

        if file_path not in result:
            result.append(file_path)

    return result


# ============================================================
# LLM CLIENT
# ============================================================

class GigaChatClient:

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        top_p: float,
    ):

        if provider not in ENDPOINTS:
            raise ValueError(
                f"Неизвестный provider: {provider}"
            )

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.url = ENDPOINTS[provider]

        self.session = requests.Session()
        self.token = self._get_token()

    def _get_token(self) -> str:

        config_path = os.path.expanduser(
            CFG[self.provider]
        )

        client = TagmeClientAdvanced(config_path)
        token = client.client.token

        if not token:
            raise RuntimeError(
                f"Не удалось получить токен для {self.provider}"
            )

        return token

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        # Payload формируется один раз.
        # При повторных отправках содержимое запроса не меняется.

        payload = {
            "model": self.model,
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
            "max_tokens": MAX_TOKENS,
            "profanity_check": False,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = self.session.post(
            self.url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=TIMEOUT,
        )

        if response.status_code == 401:

            logger.warning(
                "HTTP 401. Обновляю токен и повторяю тот же запрос."
            )

            self.token = self._get_token()

            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

            response = self.session.post(
                self.url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=TIMEOUT,
            )

        response.raise_for_status()

        data = response.json()

        try:

            content = data["choices"][0]["message"]["content"]

        except (KeyError, IndexError, TypeError) as exc:

            raise RuntimeError(
                f"Не удалось извлечь ответ модели: {data}"
            ) from exc

        if isinstance(content, list):

            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Модель вернула пустой ответ."
            )

        return content.strip()


# ============================================================
# THREAD-LOCAL CLIENT
# ============================================================

def get_client(args):

    client = getattr(
        _thread_local,
        "client",
        None,
    )

    if client is None:

        client = GigaChatClient(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        _thread_local.client = client

    return client


def reset_client():

    client = getattr(
        _thread_local,
        "client",
        None,
    )

    if client is not None:

        try:
            client.session.close()

        except Exception:
            pass

    _thread_local.client = None


# ============================================================
# RESPONSE PARSING
# ============================================================

def extract_json_array(raw_response: str) -> str:
    """Извлекает JSON-массив из ответа модели."""

    text = raw_response.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    try:

        parsed = json.loads(text)

        if isinstance(parsed, list):
            return text

    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "В ответе не найден JSON-массив."
        )

    candidate = text[start:end + 1]

    parsed = json.loads(candidate)

    if not isinstance(parsed, list):
        raise ValueError(
            "Ожидался JSON-массив."
        )

    return candidate


def parse_response(
    raw_response: str,
    expected_names: set[str],
) -> list[dict[str, str]]:
    """Проверяет структуру ответа LLM и полноту списка критериев."""

    result = json.loads(
        extract_json_array(raw_response)
    )

    if not isinstance(result, list):
        raise ValueError(
            "Ответ должен быть JSON-массивом."
        )

    normalized_result = []
    received_names = set()

    for index, item in enumerate(result):

        if not isinstance(item, dict):
            raise ValueError(
                f"Элемент {index} должен быть объектом."
            )

        for field in ("name", "answer", "mention", "reason"):

            if field not in item:
                raise ValueError(
                    f"Элемент {index}: отсутствует поле {field}."
                )

        if not isinstance(item["name"], str):
            raise ValueError(
                f"Элемент {index}: name должен быть строкой."
            )

        if not item["name"].strip():
            raise ValueError(
                f"Элемент {index}: name не должен быть пустым."
            )

        if not isinstance(item["mention"], str):
            raise ValueError(
                f"Элемент {index}: mention должен быть строкой."
            )

        if not isinstance(item["reason"], str):
            raise ValueError(
                f"Элемент {index}: reason должен быть строкой."
            )

        try:

            answer = normalize_answer(
                item["answer"]
            )

        except ValueError as exc:

            raise ValueError(
                f"Элемент {index}: {exc}"
            ) from exc

        name = normalize_and_filter_name(
            item["name"],
            f"Ответ LLM, элемент {index}",
        )

        if name is None:
            continue

        if name in received_names:
            raise ValueError(
                f"Ответ LLM содержит повторяющийся критерий: "
                f"{name!r}."
            )

        received_names.add(name)

        mention = item["mention"]

        if answer == "0" and mention.strip():
            raise ValueError(
                f"Критерий {name!r}: при answer='0' "
                "поле mention должно быть пустым."
            )

        if answer == "1" and not mention.strip():
            raise ValueError(
                f"Критерий {name!r}: при answer='1' "
                "поле mention не должно быть пустым."
            )

        normalized_result.append({
            "name": name,
            "answer": answer,
            "mention": mention,
            "reason": item["reason"],
        })

    missing_names = expected_names - received_names
    unexpected_names = received_names - expected_names

    if missing_names or unexpected_names:

        errors = []

        if missing_names:
            errors.append(
                "отсутствуют критерии: "
                + ", ".join(sorted(missing_names))
            )

        if unexpected_names:
            errors.append(
                "неизвестные критерии: "
                + ", ".join(sorted(unexpected_names))
            )

        raise ValueError(
            "Неполный или некорректный список критериев. "
            + "; ".join(errors)
        )

    return normalized_result


# ============================================================
# SYSTEM PROMPT
# ============================================================

def load_system_prompt(
    prompt_path: Path,
) -> str:
    """Загружает фактический системный промпт из файла."""

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Системный промпт не найден: {prompt_path}"
        )

    system_prompt = prompt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not system_prompt:
        raise ValueError(
            f"Системный промпт пуст: {prompt_path}"
        )

    return system_prompt


# ============================================================
# INPUT
# ============================================================

def load_input(
    input_path: Path,
) -> list[dict[str, Any]]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Ожидался JSON-массив объектов."
        )

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"Элемент {index} должен быть JSON-объектом."
            )

    return data


def validate_input(
    data: list[dict[str, Any]],
    input_field: str,
):

    if not data:
        raise ValueError(
            "Входной JSON-массив пуст."
        )

    missing_ids = [
        index
        for index, item in enumerate(data)
        if (
            not isinstance(item.get("id"), str)
            or not item["id"].strip()
        )
    ]

    if missing_ids:
        raise ValueError(
            "В объектах отсутствует корректный id. "
            f"Индексы: {missing_ids[:10]}"
        )

    duplicate_ids = set()
    seen_ids = set()

    for item in data:

        item_id = item["id"]

        if item_id in seen_ids:
            duplicate_ids.add(item_id)

        seen_ids.add(item_id)

    if duplicate_ids:
        raise ValueError(
            "Обнаружены повторяющиеся id во входном файле: "
            f"{list(duplicate_ids)[:10]}"
        )

    missing_gold = [
        index
        for index, item in enumerate(data)
        if not isinstance(item.get("result"), list)
    ]

    if missing_gold:
        raise ValueError(
            "В объектах отсутствует корректный список result. "
            f"Индексы: {missing_gold[:10]}"
        )

    missing_input = [
        index
        for index, item in enumerate(data)
        if (
            not isinstance(item.get(input_field), str)
            or not item[input_field].strip()
        )
    ]

    if missing_input:
        raise ValueError(
            f"В объектах отсутствует корректное поле "
            f"'{input_field}'. Индексы: {missing_input[:10]}"
        )


# ============================================================
# FULL PREDICTION RECORD
# ============================================================

def build_full_prediction_record(
    original_item: dict[str, Any],
    llm_record: dict[str, Any] | None,
    model: str,
    timestamp: str,
    source_files_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Сохраняет полный структурированный ответ модели."""

    prediction = None
    error = None
    raw_response = None

    if llm_record is None:

        error = "LLM result is missing"

    else:

        raw_response = llm_record.get("llm_raw") or None

        if llm_record.get("llm_error"):

            error = llm_record["llm_error"]

        elif llm_record.get("llm_result") is not None:

            prediction = llm_record["llm_result"]

        else:

            error = "LLM returned no parsed result"

    return {
        "id": original_item["id"],
        "source_files": get_source_files(
            original_item,
            source_files_map,
        ),
        "model": model,
        "timestamp": timestamp,
        "prediction": prediction,
        "raw_response": raw_response,
        "error": error,
        "attempts": (
            llm_record.get("llm_attempts", 0)
            if llm_record is not None
            else 0
        ),
    }


def append_full_predictions(
    records: list[dict[str, Any]],
    output_path: Path,
) -> int:

    return append_json_results(
        records=records,
        output_path=output_path,
    )


# ============================================================
# SINGLE RECORD EVALUATION
# ============================================================

def evaluate_record(
    row_index: int,
    user_prompt: str,
    system_prompt: str,
    args,
) -> dict[str, Any]:

    result = {
        "row_index": row_index,
        "llm_result": None,
        "llm_raw": "",
        "llm_error": "",
        "llm_attempts": 0,
    }

    # Проверяем входные критерии один раз.
    # Некорректные входные данные не исправятся повторной отправкой.

    try:

        expected_names = extract_expected_names(
            user_prompt=user_prompt,
            item_id=str(row_index),
        )

    except Exception as exc:

        result["llm_error"] = (
            f"Ошибка входных критериев: {exc}"
        )

        logger.error(
            "Объект %s | ошибка входных критериев: %s",
            row_index,
            exc,
        )

        return result

    for attempt in range(1, MAX_RETRIES + 1):

        result["llm_attempts"] = attempt

        try:

            client = get_client(args)

            raw_response = client.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            parsed_result = parse_response(
                raw_response=raw_response,
                expected_names=expected_names,
            )

            result["llm_raw"] = raw_response
            result["llm_result"] = parsed_result
            result["llm_error"] = ""

            logger.info(
                "Объект %s | успешно обработан | попыток: %s",
                row_index,
                attempt,
            )

            return result

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            result["llm_error"] = str(exc)

            logger.warning(
                "Объект %s | попытка %s/%s | ошибка: %s",
                row_index,
                attempt,
                MAX_RETRIES,
                exc,
            )

            reset_client()

            if attempt < MAX_RETRIES:

                logger.warning(
                    "Объект %s | повторная отправка через %s секунды.",
                    row_index,
                    RETRY_DELAY,
                )

                time.sleep(RETRY_DELAY)

    logger.error(
        "Объект %s | не удалось обработать после %s попыток.",
        row_index,
        MAX_RETRIES,
    )

    return result


# ============================================================
# RESULT BUILDING
# ============================================================

def build_result_record(
    original_item: dict[str, Any],
    llm_record: dict[str, Any] | None,
    model: str,
    timestamp: str,
) -> dict[str, Any]:
    """Формирует подробную запись Ground Truth и Prediction."""

    item_id = original_item["id"]

    gt_names = extract_ground_truth_names(
        original_item["result"],
        f"result объекта {item_id}",
    )

    prediction = None

    if (
        llm_record is not None
        and not llm_record.get("llm_error")
        and llm_record.get("llm_result") is not None
    ):

        predicted_names = extract_predicted_names(
            llm_record["llm_result"],
            f"prediction объекта {item_id}",
        )

        prediction = sorted(predicted_names)

    return {
        "id": item_id,
        "model": model,
        "timestamp": timestamp,
        "gt_ground_truth": sorted(gt_names),
        "prediction": prediction,
    }


def build_csv_row(
    original_item: dict[str, Any],
    llm_record: dict[str, Any] | None,
    model: str,
    timestamp: str,
) -> dict[str, Any]:
    """
    CSV содержит одну строку на диалог.

    accuracy:
        1 — полное совпадение положительных категорий;
        0 — множества отличаются;
        пусто — результат невозможно оценить.
    """

    row = {
        "timestamp": timestamp,
        "model": model,
        "id": original_item["id"],
        "accuracy": "",
    }

    try:

        gt_names = extract_ground_truth_names(
            original_item["result"],
            f"result объекта {original_item['id']}",
        )

    except (ValueError, TypeError) as exc:

        logger.warning(
            "ID %s: некорректный Ground Truth: %s",
            original_item["id"],
            exc,
        )

        return row

    if (
        llm_record is None
        or llm_record.get("llm_error")
        or llm_record.get("llm_result") is None
    ):
        return row

    try:

        predicted_names = extract_predicted_names(
            llm_record["llm_result"],
            f"prediction объекта {original_item['id']}",
        )

        row["accuracy"] = int(
            gt_names == predicted_names
        )

    except (ValueError, TypeError) as exc:

        logger.warning(
            "ID %s: некорректное предсказание: %s",
            original_item["id"],
            exc,
        )

    return row


# ============================================================
# ATOMIC JSON SAVE
# ============================================================

def atomic_save_json(
    data: Any,
    output_path: Path,
):
    """Атомарно сохраняет JSON-объект или массив."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        output_path.stem
        + ".tmp"
        + output_path.suffix
    )

    try:

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temp_path.replace(output_path)

    finally:

        if temp_path.exists():
            temp_path.unlink()


def append_json_results(
    records: list[dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Добавляет результаты в JSON.

    Уникальность:
        id + model + timestamp.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            existing = json.load(file)

        if not isinstance(existing, list):
            raise ValueError(
                f"Ожидался JSON-массив: {output_path}"
            )

    else:

        existing = []

    existing_keys = {
        (
            item.get("id"),
            item.get("model"),
            item.get("timestamp"),
        )
        for item in existing
        if isinstance(item, dict)
    }

    added = 0

    for record in records:

        key = (
            record["id"],
            record["model"],
            record["timestamp"],
        )

        if key in existing_keys:
            continue

        existing.append(record)
        existing_keys.add(key)
        added += 1

    if added:

        atomic_save_json(
            existing,
            output_path,
        )

    return added


# ============================================================
# CSV SAVE
# ============================================================

def append_csv_results(
    records: list[dict[str, Any]],
    output_path: Path,
) -> int:
    """Добавляет новые строки в CSV."""

    if not records:
        return 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        output_path.exists()
        and output_path.stat().st_size > 0
    )

    with output_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(records)

    return len(records)


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    data: list[dict[str, Any]],
    results: dict[int, dict[str, Any]],
    row_indices: list[int],
    csv_path: Path,
    json_path: Path,
    full_predictions_path: Path,
    source_files_map: dict[str, list[str]],
    model: str,
    timestamp: str,
) -> tuple[int, int, int]:

    if not row_indices:
        return 0, 0, 0

    csv_records = []
    json_records = []
    full_prediction_records = []

    for index in row_indices:

        original_item = data[index]
        llm_record = results.get(index)

        csv_records.append(
            build_csv_row(
                original_item=original_item,
                llm_record=llm_record,
                model=model,
                timestamp=timestamp,
            )
        )

        try:

            json_records.append(
                build_result_record(
                    original_item=original_item,
                    llm_record=llm_record,
                    model=model,
                    timestamp=timestamp,
                )
            )

        except (ValueError, TypeError) as exc:

            logger.warning(
                "ID %s: ошибка формирования JSON: %s",
                original_item["id"],
                exc,
            )

        full_prediction_records.append(
            build_full_prediction_record(
                original_item=original_item,
                llm_record=llm_record,
                model=model,
                timestamp=timestamp,
                source_files_map=source_files_map,
            )
        )

    csv_saved = append_csv_results(
        records=csv_records,
        output_path=csv_path,
    )

    json_saved = append_json_results(
        records=json_records,
        output_path=json_path,
    )

    full_saved = append_full_predictions(
        records=full_prediction_records,
        output_path=full_predictions_path,
    )

    return csv_saved, json_saved, full_saved


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    data: list[dict[str, Any]],
    results: dict[int, dict[str, Any]],
) -> dict[str, Any]:

    evaluated = 0
    correct = 0

    invalid_gold = 0
    failed = 0

    for index, llm_record in results.items():

        original_item = data[index]

        try:

            gt_names = extract_ground_truth_names(
                original_item["result"],
                f"result объекта {original_item['id']}",
            )

        except (ValueError, TypeError):

            invalid_gold += 1
            continue

        if (
            llm_record.get("llm_error")
            or llm_record.get("llm_result") is None
        ):

            failed += 1
            continue

        try:

            predicted_names = extract_predicted_names(
                llm_record["llm_result"],
                f"prediction объекта {original_item['id']}",
            )

        except (ValueError, TypeError):

            failed += 1
            continue

        evaluated += 1

        if gt_names == predicted_names:
            correct += 1

    total_records = len(data)
    processed_records = len(results)

    accuracy = (
        correct / evaluated
        if evaluated
        else None
    )

    # Доля объектов, для которых удалось получить
    # валидный результат и провести сравнение с GT.
    coverage = (
        evaluated / total_records
        if total_records
        else None
    )

    # Доля объектов, для которых завершилась обработка.
    processing_coverage = (
        processed_records / total_records
        if total_records
        else None
    )

    records_without_attempts = sum(
        1
        for result in results.values()
        if result.get("llm_attempts", 0) == 0
    )

    return {
        "total_records": total_records,
        "processed_records": processed_records,
        "evaluated_records": evaluated,
        "correct_records": correct,
        "failed_records": failed,
        "invalid_gold_records": invalid_gold,
        "records_without_llm_attempts": records_without_attempts,
        "coverage": coverage,
        "processing_coverage": processing_coverage,
        "exact_match_accuracy": accuracy,
    }


def log_metrics(metrics: dict[str, Any]):

    logger.info("=" * 70)
    logger.info("EXACT MATCH ACCURACY")
    logger.info("=" * 70)

    logger.info(
        "Всего объектов: %s",
        metrics["total_records"],
    )

    logger.info(
        "Обработано: %s",
        metrics["processed_records"],
    )

    logger.info(
        "Оценено: %s",
        metrics["evaluated_records"],
    )

    logger.info(
        "Корректных совпадений: %s",
        metrics["correct_records"],
    )

    logger.info(
        "Ошибок LLM: %s",
        metrics["failed_records"],
    )

    logger.info(
        "Некорректная разметка: %s",
        metrics["invalid_gold_records"],
    )

    logger.info(
        "Без попыток обращения к LLM: %s",
        metrics["records_without_llm_attempts"],
    )

    accuracy = metrics["exact_match_accuracy"]

    if accuracy is None:

        logger.info(
            "Exact Match Accuracy: N/A"
        )

    else:

        logger.info(
            "Exact Match Accuracy: %.4f (%s/%s)",
            accuracy,
            metrics["correct_records"],
            metrics["evaluated_records"],
        )

    coverage = metrics["coverage"]

    if coverage is not None:

        logger.info(
            "Evaluation coverage: %.4f",
            coverage,
        )

    processing_coverage = metrics["processing_coverage"]

    if processing_coverage is not None:

        logger.info(
            "Processing coverage: %.4f",
            processing_coverage,
        )

    logger.info("=" * 70)


# ============================================================
# RUN INITIALIZATION
# ============================================================

def initialize_run(
    args,
    input_path: Path,
    prompt_path: Path,
    source_files_map_path: Path | None,
    data: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    """Создаёт директорию и метаданные нового запуска."""

    run_id = create_run_id()

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Все результаты конкретного запуска
    # сохраняются внутри его собственной директории.

    csv_path = run_dir / Path(args.output).name
    json_path = run_dir / Path(args.json_output).name

    full_predictions_path = (
        run_dir / Path(args.full_predictions_output).name
    )

    metrics_path = (
        run_dir / Path(args.metrics_output).name
    )

    metadata_path = run_dir / "run.json"

    started_at = get_timestamp()

    dataset_hash = get_file_hash(input_path)
    system_prompt_hash = get_text_hash(system_prompt)

    source_files_map_hash = (
        get_file_hash(source_files_map_path)
        if source_files_map_path is not None
        else None
    )

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,

        "model": args.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": MAX_TOKENS,

        "workers": args.workers,
        "max_retries": MAX_RETRIES,
        "retry_delay": RETRY_DELAY,
        "checkpoint_every": CHECKPOINT_EVERY,

        "dataset_path": str(input_path.resolve()),
        "dataset_hash": dataset_hash,

        "system_prompt_file": str(prompt_path.resolve()),
        "system_prompt_hash": system_prompt_hash,

        "source_files_map": (
            str(source_files_map_path.resolve())
            if source_files_map_path
            else None
        ),
        "source_files_map_hash": source_files_map_hash,

        "input_field": args.input_field,
        "total_records": len(data),

        "processed_records": 0,
        "evaluated_records": 0,
        "correct_records": 0,
        "failed_records": 0,
        "invalid_gold_records": 0,

        "coverage": 0.0,
        "processing_coverage": 0.0,
        "exact_match_accuracy": None,

        "error": None,

        "run_directory": str(run_dir.resolve()),
        "results_path": str(json_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "full_predictions_path": str(
            full_predictions_path.resolve()
        ),
    }

    registry_entry = {
        field: metadata.get(field)
        for field in REGISTRY_FIELDS
    }

    save_run_metadata(
        metadata,
        metadata_path,
    )

    save_registry_entry(
        registry_entry,
    )

    logger.info("=" * 70)
    logger.info("СОЗДАН НОВЫЙ ЗАПУСК")
    logger.info("=" * 70)

    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Dataset SHA-256: %s", dataset_hash)
    logger.info("System prompt SHA-256: %s", system_prompt_hash)

    logger.info(
        "Реестр запусков: %s",
        REGISTRY_PATH.resolve(),
    )

    logger.info("=" * 70)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "registry_entry": registry_entry,
        "started_at": started_at,
        "csv_path": csv_path,
        "json_path": json_path,
        "full_predictions_path": full_predictions_path,
        "metrics_path": metrics_path,
    }


# ============================================================
# RUN FINALIZATION
# ============================================================

def finalize_run(
    run: dict[str, Any],
    results: dict[int, dict[str, Any]],
    data: list[dict[str, Any]],
    status: str,
    error: str | None = None,
):
    """Фиксирует финальные метрики и статус запуска."""

    metrics = calculate_metrics(
        data=data,
        results=results,
    )

    finished_at = get_timestamp()

    started_datetime = datetime.fromisoformat(
        run["started_at"]
    )

    finished_datetime = datetime.fromisoformat(
        finished_at
    )

    duration_seconds = round(
        (
            finished_datetime - started_datetime
        ).total_seconds(),
        2,
    )

    metrics.update({
        "run_id": run["run_id"],
        "status": status,
        "started_at": run["started_at"],
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "model": run["metadata"]["model"],
        "provider": run["metadata"]["provider"],
        "temperature": run["metadata"]["temperature"],
        "top_p": run["metadata"]["top_p"],
        "dataset_hash": run["metadata"]["dataset_hash"],
        "system_prompt_hash": run["metadata"]["system_prompt_hash"],
        "error": error,
    })

    atomic_save_json(
        metrics,
        run["metrics_path"],
    )

    updates = {
        "status": status,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "error": error,
    }

    updates.update(
        get_tracking_metrics(metrics)
    )

    update_run_tracking(
        metadata=run["metadata"],
        metadata_path=run["metadata_path"],
        registry_entry=run["registry_entry"],
        updates=updates,
    )

    log_metrics(metrics)

    logger.info("=" * 70)
    logger.info("ЗАПУСК ЗАВЕРШЁН")
    logger.info("=" * 70)

    logger.info("Run ID: %s", run["run_id"])
    logger.info("Status: %s", status)
    logger.info("Duration: %s seconds", duration_seconds)

    logger.info(
        "Результаты: %s",
        run["json_path"].resolve(),
    )

    logger.info(
        "CSV: %s",
        run["csv_path"].resolve(),
    )

    logger.info(
        "Полные предсказания: %s",
        run["full_predictions_path"].resolve(),
    )

    logger.info(
        "Метрики: %s",
        run["metrics_path"].resolve(),
    )

    logger.info(
        "Метаданные: %s",
        run["metadata_path"].resolve(),
    )

    logger.info(
        "Общий реестр: %s",
        REGISTRY_PATH.resolve(),
    )

    logger.info("=" * 70)


# ============================================================
# CHECKPOINT TRACKING
# ============================================================

def save_run_checkpoint(
    run: dict[str, Any],
    data: list[dict[str, Any]],
    results: dict[int, dict[str, Any]],
    row_indices: list[int],
    written_indices: set[int],
    source_files_map: dict[str, list[str]],
):
    """Сохраняет результаты и обновляет промежуточные метрики."""

    if not row_indices:
        return 0, 0, 0

    (
        csv_saved,
        json_saved,
        full_saved,
    ) = save_checkpoint(
        data=data,
        results=results,
        row_indices=row_indices,
        csv_path=run["csv_path"],
        json_path=run["json_path"],
        full_predictions_path=run["full_predictions_path"],
        source_files_map=source_files_map,
        model=run["metadata"]["model"],
        timestamp=run["started_at"],
    )

    written_indices.update(row_indices)

    metrics = calculate_metrics(
        data=data,
        results=results,
    )

    metrics.update({
        "run_id": run["run_id"],
        "status": "running",
        "started_at": run["started_at"],
        "model": run["metadata"]["model"],
        "provider": run["metadata"]["provider"],
    })

    atomic_save_json(
        metrics,
        run["metrics_path"],
    )

    updates = get_tracking_metrics(metrics)

    update_run_tracking(
        metadata=run["metadata"],
        metadata_path=run["metadata_path"],
        registry_entry=run["registry_entry"],
        updates=updates,
    )

    return csv_saved, json_saved, full_saved


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(args):

    input_path = Path(args.input)
    prompt_path = Path(args.system_prompt_file)

    source_files_map_path = (
        Path(args.source_files_map)
        if args.source_files_map
        else None
    )

    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    data = load_input(input_path)

    validate_input(
        data=data,
        input_field=args.input_field,
    )

    source_files_map = load_source_files_map(
        source_files_map_path
    )

    system_prompt = load_system_prompt(
        prompt_path
    )

    total_rows = len(data)

    # --------------------------------------------------------
    # INITIALIZE RUN
    # --------------------------------------------------------

    run = initialize_run(
        args=args,
        input_path=input_path,
        prompt_path=prompt_path,
        source_files_map_path=source_files_map_path,
        data=data,
        system_prompt=system_prompt,
    )

    logger.info("=" * 70)
    logger.info("ЗАПУСК ОЦЕНКИ")
    logger.info("=" * 70)

    logger.info("Входной файл: %s", input_path.resolve())
    logger.info("Системный промпт: %s", prompt_path.resolve())

    logger.info(
        "Выходной CSV: %s",
        run["csv_path"].resolve(),
    )

    logger.info(
        "Выходной JSON: %s",
        run["json_path"].resolve(),
    )

    logger.info(
        "Полные ответы модели: %s",
        run["full_predictions_path"].resolve(),
    )

    logger.info(
        "Итоговые метрики: %s",
        run["metrics_path"].resolve(),
    )

    logger.info(
        "Файл связей исходных файлов: %s",
        source_files_map_path.resolve()
        if source_files_map_path
        else "не задан",
    )

    logger.info("Количество объектов: %s", total_rows)
    logger.info("Поле для LLM: %s", args.input_field)
    logger.info("Provider: %s", args.provider)
    logger.info("Модель: %s", args.model)
    logger.info("Workers: %s", args.workers)
    logger.info("Run ID: %s", run["run_id"])

    # --------------------------------------------------------
    # EVALUATION STATE
    # --------------------------------------------------------

    results: dict[int, dict[str, Any]] = {}

    futures = {}
    written_indices: set[int] = set()

    completed = 0
    successful = 0
    failed = 0

    executor = ThreadPoolExecutor(
        max_workers=args.workers
    )

    # --------------------------------------------------------
    # SUBMIT TASKS
    # --------------------------------------------------------

    try:

        for index, item in enumerate(data):

            user_prompt = item.get(
                args.input_field
            )

            future = executor.submit(
                evaluate_record,
                index,
                user_prompt,
                system_prompt,
                args,
            )

            futures[future] = index

        # ----------------------------------------------------
        # COLLECT RESULTS
        # ----------------------------------------------------

        for future in as_completed(futures):

            index = futures[future]

            try:

                result = future.result()

            except Exception as exc:

                result = {
                    "row_index": index,
                    "llm_result": None,
                    "llm_raw": "",
                    "llm_error": str(exc),
                    "llm_attempts": 0,
                }

                logger.exception(
                    "Объект %s | ошибка выполнения задачи.",
                    index,
                )

            results[index] = result
            completed += 1

            if result["llm_error"]:
                failed += 1
            else:
                successful += 1

            # ------------------------------------------------
            # PERIODIC CHECKPOINT
            # ------------------------------------------------

            if completed % CHECKPOINT_EVERY == 0:

                pending_indices = sorted(
                    set(results.keys()) - written_indices
                )

                (
                    csv_saved,
                    json_saved,
                    full_saved,
                ) = save_run_checkpoint(
                    run=run,
                    data=data,
                    results=results,
                    row_indices=pending_indices,
                    written_indices=written_indices,
                    source_files_map=source_files_map,
                )

                logger.info(
                    "Прогресс: %s/%s | Успешно: %s | "
                    "Ошибки: %s | CSV: +%s | JSON: +%s | "
                    "Full: +%s",
                    completed,
                    total_rows,
                    successful,
                    failed,
                    csv_saved,
                    json_saved,
                    full_saved,
                )

        # ----------------------------------------------------
        # NORMAL SHUTDOWN
        # ----------------------------------------------------

        executor.shutdown(
            wait=True,
        )

        # ----------------------------------------------------
        # FINAL CHECKPOINT
        # ----------------------------------------------------

        pending_indices = sorted(
            set(results.keys()) - written_indices
        )

        (
            csv_saved,
            json_saved,
            full_saved,
        ) = save_run_checkpoint(
            run=run,
            data=data,
            results=results,
            row_indices=pending_indices,
            written_indices=written_indices,
            source_files_map=source_files_map,
        )

        logger.info(
            "Финальное сохранение: CSV +%s | JSON +%s | Full +%s",
            csv_saved,
            json_saved,
            full_saved,
        )

        # ----------------------------------------------------
        # DETERMINE FINAL STATUS
        # ----------------------------------------------------

        metrics = calculate_metrics(
            data=data,
            results=results,
        )

        has_errors = (
            metrics["failed_records"] > 0
            or metrics["invalid_gold_records"] > 0
            or metrics["processed_records"] < metrics["total_records"]
        )

        final_status = (
            "completed_with_errors"
            if has_errors
            else "completed"
        )

        finalize_run(
            run=run,
            results=results,
            data=data,
            status=final_status,
        )

    # --------------------------------------------------------
    # INTERRUPTED RUN
    # --------------------------------------------------------

    except KeyboardInterrupt:

        logger.warning(
            "Получен сигнал прерывания. Завершаю обработку задач."
        )

        # Отменяем задачи, которые ещё не начали выполняться.
        for future in futures:
            future.cancel()

        # Дожидаемся завершения уже выполняющихся задач.
        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        # Собираем результаты всех задач, которые успели завершиться.
        for future, index in futures.items():

            if index in results:
                continue

            if future.cancelled() or not future.done():
                continue

            try:

                results[index] = future.result()

            except Exception as exc:

                results[index] = {
                    "row_index": index,
                    "llm_result": None,
                    "llm_raw": "",
                    "llm_error": str(exc),
                    "llm_attempts": 0,
                }

        # Сохраняем всё, что успело завершиться.
        pending_indices = sorted(
            set(results.keys()) - written_indices
        )

        try:

            save_run_checkpoint(
                run=run,
                data=data,
                results=results,
                row_indices=pending_indices,
                written_indices=written_indices,
                source_files_map=source_files_map,
            )

        except Exception:

            logger.exception(
                "Не удалось полностью сохранить результаты "
                "после прерывания."
            )

        try:

            finalize_run(
                run=run,
                results=results,
                data=data,
                status="interrupted",
                error="Run interrupted by user",
            )

        except Exception:

            logger.exception(
                "Не удалось обновить метаданные прерванного запуска."
            )

        raise

    # --------------------------------------------------------
    # FAILED RUN
    # --------------------------------------------------------

    except Exception as exc:

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        logger.exception(
            "Критическая ошибка во время запуска."
        )

        try:

            # Сохраняем уже полученные результаты.
            pending_indices = sorted(
                set(results.keys()) - written_indices
            )

            if pending_indices:

                save_run_checkpoint(
                    run=run,
                    data=data,
                    results=results,
                    row_indices=pending_indices,
                    written_indices=written_indices,
                    source_files_map=source_files_map,
                )

            finalize_run(
                run=run,
                results=results,
                data=data,
                status="failed",
                error=str(exc),
            )

        except Exception:

            logger.exception(
                "Не удалось сохранить финальное состояние "
                "аварийного запуска."
            )

        raise


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Многопоточная оценка JSON-диалогов "
            "с расчётом Exact Match Accuracy "
            "и сохранением истории экспериментальных запусков."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Путь к входному prepared JSON.",
    )

    parser.add_argument(
        "--output",
        default="results.csv",
        help="Имя итогового CSV внутри директории запуска.",
    )

    parser.add_argument(
        "--json-output",
        default="results.json",
        help="Имя подробного JSON внутри директории запуска.",
    )

    parser.add_argument(
        "--full-predictions-output",
        default="predictions_full.json",
        help="Имя JSON с полными ответами модели.",
    )

    parser.add_argument(
        "--metrics-output",
        default="metrics.json",
        help="Имя JSON с метриками.",
    )

    parser.add_argument(
        "--source-files-map",
        default=None,
        help=(
            "JSON-файл со связями id -> список исходных файлов. "
            "Формат: {\"12345\": [\"part_01.json\"]}."
        ),
    )

    parser.add_argument(
        "--system-prompt-file",
        default="system_prompt.txt",
        help="Путь к файлу системного промпта.",
    )

    parser.add_argument(
        "--input-field",
        default=INPUT_FIELD,
        help="Поле JSON с текстом для LLM.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=5,
        help="Количество потоков.",
    )

    parser.add_argument(
        "--provider",
        choices=["aihf", "direct", "sigma"],
        default="aihf",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error(
            "--workers должен быть не меньше 1."
        )

    if not 0 <= args.temperature <= 2:
        parser.error(
            "--temperature должен быть в диапазоне 0..2."
        )

    if not 0 < args.top_p <= 1:
        parser.error(
            "--top-p должен быть в диапазоне (0, 1]."
        )

    return args


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    arguments = parse_args()

    try:

        run_evaluation(arguments)

    except KeyboardInterrupt:

        logger.warning(
            "Обработка прервана пользователем."
        )

    except Exception:

        logger.exception(
            "Критическая ошибка."
        )

        raise



# python evaluate.py \
#     --input ./predicts/pool_1_prepared.json \
#     --system-prompt-file prompts/system_prompt.md \
#     --provider aihf \
#     --model DeepSeek-V4.1-Flash \
#     --workers 20