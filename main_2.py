
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
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Ожидался JSON-массив: {path}")

    return data


def build_part_system_prompt(item: dict[str, Any]) -> str:
    """Формирует текстовый контекст из диалога и классов."""

    dialogue = (
        item.get("data", {}).get("dialogue", "")
    )

    classes = (
        item.get("premarkup", {}).get("classes", [])
    )

    class_names = [
        " ".join(str(cls["name"]).split())
        for cls in classes
        if isinstance(cls, dict) and cls.get("name")
    ]

    variants = "\n".join(
        f"- {name}"
        for index, name in enumerate(class_names, start=1)
    )

    return f'''**Диалог между сотрудником банка и клиентом**

```
{dialogue}
```


**Варианты выводов из диалога:**

{variants}
'''


def extract_positive_results(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """Оставляет только критерии с answer == 1."""

    output_criteria = (
        item.get("result", {}).get("output_criteria", [])
    )

    if not isinstance(output_criteria, list):
        return []

    results = []

    for criterion in output_criteria:
        if not isinstance(criterion, dict):
            continue

        answer = str(criterion.get("answer", "")).strip()

        if answer != "1":
            continue

        results.append({
            "name": criterion.get("name", ""),
            "answer": answer,
            "comment": criterion.get("comment", ""),
            "mention": criterion.get("mention", ""),
        })

    return results


def extract_results(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """Извлекает все критерии без фильтрации по answer."""

    output_criteria = (
        item.get("result", {}).get("output_criteria", [])
    )

    if not isinstance(output_criteria, list):
        return []

    results = []

    for criterion in output_criteria:
        if not isinstance(criterion, dict):
            continue

        results.append({
            "name": criterion.get("name", ""),
            "answer": str(criterion.get("answer", "")),
            "comment": criterion.get("comment", ""),
            "mention": criterion.get("mention", ""),
        })

    return results



def transform_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    """Преобразует исходный объект в целевую структуру."""

    return {
        "part_system_prompt": build_part_system_prompt(item),
        "result": extract_results(item),
    }


def process_pool(
    input_path: Path,
    output_path: Path,
) -> None:

    if not input_path.exists():
        logger.error("Файл не найден: %s", input_path)
        return

    logger.info("Обработка файла: %s", input_path)

    source_data = load_json(input_path)

    transformed_data = [
        transform_item(item)
        for item in source_data
    ]

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

    total_results = sum(
        len(item["result"])
        for item in transformed_data
    )

    logger.info("Исходных объектов: %d", len(source_data))
    logger.info("Преобразовано объектов: %d", len(transformed_data))
    logger.info("Положительных результатов: %d", total_results)
    logger.info("Результат сохранён: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Формирование датасета из объединённых пулов."
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
