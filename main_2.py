
from pathlib import Path
import argparse
import json
import logging
import math

from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict[str, Any]]:
    """Загружает JSON-массив и проверяет его структуру."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Ожидался JSON-массив: {path}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Объект с индексом {index} должен быть словарём"
            )

    return data


def normalize_name(value: Any) -> str | None:
    """
    Проверяет название критерия.

    Сохраняет исходные пробелы и переносы строк,
    чтобы название не отличалось от исходного значения.
    """

    if not isinstance(value, str):
        return None

    if not value.strip():
        return None

    if value.strip().lower() == "fork":
        return None

    return value


def normalize_answer(value: Any) -> str:
    """
    Приводит answer к строковому значению "1" или "0".

    Поддерживает:
    - 1 / 0;
    - 1.0 / 0.0;
    - "1" / "0";
    - "1.0" / "0.0";
    - True / False.
    """

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Некорректное значение answer: {value}")

        if value == 1:
            return "1"

        if value == 0:
            return "0"

    if isinstance(value, str):
        normalized = value.strip()

        if normalized in {"1", "1.0"}:
            return "1"

        if normalized in {"0", "0.0"}:
            return "0"

    raise ValueError(f"Неизвестное значение answer: {value!r}")


def get_dialogue(item: dict[str, Any]) -> str:
    """Извлекает текст диалога."""

    data = item.get("data") or {}

    if not isinstance(data, dict):
        return ""

    dialogue = data.get("dialogue", "")

    if dialogue is None:
        return ""

    if not isinstance(dialogue, str):
        dialogue = str(dialogue)

    return dialogue


def extract_unique_class_names(
    item: dict[str, Any],
) -> list[str]:
    """Извлекает уникальные названия критериев из premarkup.classes."""

    premarkup = item.get("premarkup") or {}

    if not isinstance(premarkup, dict):
        return []

    classes = premarkup.get("classes", [])

    if not isinstance(classes, list):
        return []

    names = []
    seen = set()

    for cls in classes:
        if not isinstance(cls, dict):
            continue

        name = normalize_name(cls.get("name"))

        if name is None or name in seen:
            continue

        seen.add(name)
        names.append(name)

    return names


def build_part_system_prompt(
    item: dict[str, Any],
) -> str:
    """
    Формирует входной JSON для модели.

    Структура соответствует системному промпту:
    {
        "data": {
            "dialogue": "..."
        },
        "premarkup": {
            "classes": [
                {"name": "..."}
            ]
        }
    }
    """

    dialogue = get_dialogue(item)
    class_names = extract_unique_class_names(item)

    input_data = {
        "data": {
            "dialogue": dialogue,
        },
        "premarkup": {
            "classes": [
                {"name": name}
                for name in class_names
            ],
        },
    }

    return json.dumps(
        input_data,
        ensure_ascii=False,
        indent=2,
    )


def extract_results(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Извлекает все эталонные критерии.

    Сохраняет как положительные, так и отрицательные ответы,
    поскольку оценка выполняется по каждому критерию.
    """

    source_result = item.get("result") or {}

    if not isinstance(source_result, dict):
        return []

    output_criteria = source_result.get("output_criteria", [])

    if not isinstance(output_criteria, list):
        return []

    results = []
    seen = set()

    for index, criterion in enumerate(output_criteria):
        if not isinstance(criterion, dict):
            logger.warning(
                "Пропущен некорректный эталонный критерий: index=%d",
                index,
            )
            continue

        name = normalize_name(criterion.get("name"))

        if name is None:
            continue

        if name in seen:
            logger.warning(
                "Обнаружен дублирующийся критерий в result: %s",
                name,
            )
            continue

        seen.add(name)

        try:
            answer = normalize_answer(criterion.get("answer"))
        except ValueError:
            logger.warning(
                "Некорректный answer у критерия '%s': %r",
                name,
                criterion.get("answer"),
            )
            continue

        reason = criterion.get("reason")

        if reason is None:
            reason = criterion.get("comment", "")

        mention = criterion.get("mention", "")

        results.append({
            "name": name,
            "answer": answer,
            "mention": str(mention or ""),
            "reason": str(reason or ""),
        })

    return results


def transform_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    """Преобразует исходный объект в целевую структуру."""

    filename = item.get("filename", "")

    item_id = (
        Path(filename).stem
        if isinstance(filename, str) and filename.strip()
        else ""
    )

    return {
        "id": item_id,
        "part_system_prompt": build_part_system_prompt(item),
        "result": extract_results(item),
    }


def process_pool(
    input_path: Path,
    output_path: Path,
) -> None:
    """Обрабатывает один пул и сохраняет подготовленный датасет."""

    if not input_path.exists():
        logger.warning(
            "Файл не найден, пропускаем: %s",
            input_path,
        )
        return

    logger.info("Обработка файла: %s", input_path)

    source_data = load_json(input_path)

    transformed_data = []

    total_classes = 0
    total_results = 0
    positive_results = 0
    negative_results = 0
    empty_dialogues = 0
    mismatched_items = 0

    for index, item in enumerate(source_data):
        filename = item.get("filename")

        if not isinstance(filename, str) or not filename.strip():
            logger.warning(
                "Пропущен объект с индексом %d: отсутствует filename",
                index,
            )
            continue

        item_id = Path(filename).stem

        if not item_id:
            logger.warning(
                "Пропущен объект с индексом %d: некорректный filename",
                index,
            )
            continue

        dialogue = get_dialogue(item)

        if not dialogue.strip():
            empty_dialogues += 1

        class_names = extract_unique_class_names(item)
        results = extract_results(item)

        class_name_set = set(class_names)
        result_name_set = {
            result["name"]
            for result in results
        }

        missing_results = class_name_set - result_name_set
        extra_results = result_name_set - class_name_set

        if missing_results or extra_results:
            mismatched_items += 1

            logger.warning(
                "Несоответствие критериев для id=%s: "
                "отсутствуют в result=%d, лишние в result=%d",
                item_id,
                len(missing_results),
                len(extra_results),
            )

        transformed_item = transform_item(item)

        total_classes += len(class_names)
        total_results += len(results)

        positive_results += sum(
            result["answer"] == "1"
            for result in results
        )

        negative_results += sum(
            result["answer"] == "0"
            for result in results
        )

        transformed_data.append(transformed_item)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            transformed_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("-" * 60)
    logger.info("Исходных объектов: %d", len(source_data))
    logger.info("Преобразовано объектов: %d", len(transformed_data))
    logger.info(
        "Пропущено объектов: %d",
        len(source_data) - len(transformed_data),
    )
    logger.info("Диалогов без текста: %d", empty_dialogues)
    logger.info("Объектов с несовпадением критериев: %d", mismatched_items)
    logger.info("Уникальных входных критериев: %d", total_classes)
    logger.info("Всего эталонных критериев: %d", total_results)
    logger.info("Положительных критериев (answer=1): %d", positive_results)
    logger.info("Отрицательных критериев (answer=0): %d", negative_results)

    if total_results:
        logger.info(
            "Доля положительных критериев: %.2f%%",
            positive_results / total_results * 100,
        )

    logger.info("Результат сохранён: %s", output_path)
    logger.info("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Формирование подготовленных датасетов "
            "из объединённых пулов."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Директория с файлами pool_N_merged.json.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Директория для итоговых файлов.",
    )

    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Номера пулов для обработки.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    for pool_number in args.pools:
        input_path = (
            args.input_dir / f"pool_{pool_number}_merged.json"
        )

        output_path = (
            output_dir / f"pool_{pool_number}_prepared.json"
        )

        process_pool(
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
