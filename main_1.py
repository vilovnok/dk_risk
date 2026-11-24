
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


def build_best_results_index(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    best_results = {}

    for item in results:
        filename = item.get("file_name")
        quality = item.get("quality")

        if not filename or quality is None:
            continue

        try:
            quality = float(quality)
        except (TypeError, ValueError):
            logger.warning(
                "Некорректный quality для %s: %s",
                filename,
                quality,
            )
            continue

        current_best = best_results.get(filename)

        if current_best is None or quality > current_best["quality"]:
            best_results[filename] = {
                "quality": quality,
                "result": item.get("result"),
            }

    return best_results


def merge_arrays(
    source_data: list[dict[str, Any]],
    results_data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:

    best_results = build_best_results_index(results_data)

    merged_data = []
    matched = 0
    unmatched = 0

    for item in source_data:
        merged_item = item.copy()
        filename = merged_item.get("filename")

        if filename in best_results:
            best_result = best_results[filename]

            merged_item["quality"] = best_result["quality"]
            merged_item["result"] = best_result["result"]

            matched += 1
        else:
            unmatched += 1

        merged_data.append(merged_item)

    return merged_data, matched, unmatched


def process_pool(
    source_path: Path,
    results_path: Path,
    output_path: Path,
) -> None:

    logger.info("Обработка пары: %s + %s", source_path.name, results_path.name)

    if not source_path.exists():
        logger.error("Исходный файл не найден: %s", source_path)
        return

    if not results_path.exists():
        logger.error("Файл результатов не найден: %s", results_path)
        return

    source_data = load_json(source_path)
    results_data = load_json(results_path)

    merged_data, matched, unmatched = merge_arrays(
        source_data,
        results_data,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            merged_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Исходных объектов: %d", len(source_data))
    logger.info("Сопоставлено: %d", matched)
    logger.info("Без соответствия: %d", unmatched)
    logger.info("Результат сохранён: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Объединение трёх пар JSON-файлов."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Директория с исходными файлами.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Директория для результатов. По умолчанию input-dir.",
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
        source_path = args.input_dir / f"pool_{pool_number}.json"
        results_path = args.input_dir / f"pool_{pool_number}_gt.json"
        output_path = output_dir / f"pool_{pool_number}_merged.json"

        process_pool(
            source_path,
            results_path,
            output_path,
        )


if __name__ == "__main__":
    main()


