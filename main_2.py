from pathlib import Path
import argparse
import json
import logging

from pathlib import Path
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
    """Нормализует name, не изменяя его формулировку."""

    if not isinstance(value, str):
        return None

    name = " ".join(value.split())

    if not name:
        return None

    if name.lower() == "fork":
        return None

    return name


def is_positive_answer(value: Any) -> bool:
    """Проверяет, является ли ответ положительным."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value == 1

    if isinstance(value, str):
        return value.strip() in {"1", "1.0"}

    return False


def extract_unique_class_names(
    item: dict[str, Any],
) -> list[str]:
    """Извлекает уникальные name из premarkup.classes."""

    premarkup = item.get("premarkup") or {}
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
    """Формирует текстовый контекст из диалога и уникальных классов."""

    data = item.get("data") or {}
    dialogue = data.get("dialogue", "")

    if not isinstance(dialogue, str):
        dialogue = str(dialogue)

    class_names = extract_unique_class_names(item)

    variants = "\n".join(
        f"- {name}"
        for name in class_names
    )

    return f"""**Диалог между сотрудником банка и клиентом**

```
{dialogue}
```


**Варианты выводов из диалога:**

{variants}
"""


def extract_positive_results(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """Оставляет только положительные критерии, исключая fork."""

    output_criteria = (
        (item.get("result") or {}).get("output_criteria", [])
    )

    if not isinstance(output_criteria, list):
        return []

    results = []
    seen = set()

    for criterion in output_criteria:
        if not isinstance(criterion, dict):
            continue

        if not is_positive_answer(criterion.get("answer")):
            continue

        name = normalize_name(criterion.get("name"))

        if name is None or name in seen:
            continue

        seen.add(name)

        results.append({
            "name": name,
            "answer": "1",
            "comment": str(criterion.get("comment") or ""),
            "mention": str(criterion.get("mention") or ""),
        })

    return results


def extract_results(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """Извлекает все критерии, исключая fork и дублирующиеся name."""

    output_criteria = (
        (item.get("result") or {}).get("output_criteria", [])
    )

    if not isinstance(output_criteria, list):
        return []

    results = []
    seen = set()

    for criterion in output_criteria:
        if not isinstance(criterion, dict):
            continue

        name = normalize_name(criterion.get("name"))

        if name is None or name in seen:
            continue

        seen.add(name)

        results.append({
            "name": name,
            "answer": str(criterion.get("answer", "")),
            "comment": str(criterion.get("comment") or ""),
            "mention": str(criterion.get("mention") or ""),
        })

    return results


def transform_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    """Преобразует исходный объект в целевую структуру."""

    filename = item.get("filename", "")

    return {
        "id": Path(filename).stem if filename else "",
        "part_system_prompt": build_part_system_prompt(item),
        "result": extract_results(item),
    }


def process_pool(
    input_path: Path,
    output_path: Path,
) -> None:
    """Обрабатывает один пул и сохраняет подготовленный датасет."""

    if not input_path.exists():
        logger.warning("Файл не найден, пропускаем: %s", input_path)
        return

    logger.info("Обработка файла: %s", input_path)

    source_data = load_json(input_path)

    transformed_data = []
    total_classes = 0
    total_results = 0
    positive_results = 0
    empty_dialogues = 0

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

        dialogue = (item.get("data") or {}).get("dialogue", "")

        if not isinstance(dialogue, str) or not dialogue.strip():
            empty_dialogues += 1

        transformed_item = transform_item(item)

        class_names = extract_unique_class_names(item)
        results = transformed_item["result"]

        total_classes += len(class_names)
        total_results += len(results)
        positive_results += sum(
            result["answer"] == "1"
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
    logger.info("Пропущено объектов: %d", len(source_data) - len(transformed_data))
    logger.info("Диалогов без текста: %d", empty_dialogues)
    logger.info("Уникальных классов: %d", total_classes)
    logger.info("Всего критериев: %d", total_results)
    logger.info("Положительных критериев: %d", positive_results)
    logger.info("Результат сохранён: %s", output_path)
    logger.info("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Формирование подготовленных датасетов из объединённых пулов."
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
