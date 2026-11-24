import argparse
import hashlib
import json
import logging
import sys
import uuid

from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from classifier.embedder import MentionEmbedder
from classifier.evaluator import DatasetEvaluator
from classifier.llm import LLMClient
from classifier.metrics import calculate_metrics
from classifier.tracker import ExperimentTracker

from config import (
    DATA_DIR,
    DATABASE_PATH,
    DEFAULT_MODEL,
    DEFAULT_WORKERS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    MENTION_THRESHOLD,
    OUTPUT_DIR,
    RUNS_DIR,
    TEMPERATURE,
    TOP_P,
)

from prompts.sales_evaluation import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Многопоточная оценка моделей для проекта продаж ДК."
    )

    parser.add_argument(
        "--provider",
        choices=["aihf", "direct"],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--experiment-name",
        default="debit-card-evaluation",
    )

    parser.add_argument(
        "--pools",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Максимум одновременно отправленных запросов.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
    )

    parser.add_argument(
        "--mention-threshold",
        type=float,
        default=MENTION_THRESHOLD,
    )

    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=EMBEDDING_DEVICE,
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Явно задать ID нового эксперимента.",
    )

    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Продолжить существующий эксперимент.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
    )

    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Ожидался JSON-массив: {path}"
        )

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(
                f"Запись {index} должна быть объектом."
            )

        if not isinstance(record.get("part_system_prompt"), str):
            raise ValueError(
                f"Некорректный part_system_prompt: запись {index}"
            )

        if not isinstance(record.get("result"), list):
            raise ValueError(
                f"Некорректный result: запись {index}"
            )

        for criterion in record["result"]:
            if not all(
                key in criterion
                for key in ("name", "answer", "comment", "mention")
            ):
                raise ValueError(
                    f"Некорректный критерий: запись {index}"
                )

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(path: Path, data: Any):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def build_run_config(args, dataset_hashes):
    prompt_hash = hashlib.sha256(
        SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()

    return {
        "provider": args.provider,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "embedding_model": args.embedding_model,
        "embedding_device": args.device,
        "embedding_batch_size": args.embedding_batch_size,
        "mention_threshold": args.mention_threshold,
        "system_prompt_sha256": prompt_hash,
        "dataset_hashes": dataset_hashes,
        "pools": args.pools,
        "limit": args.limit,
        "workers": args.workers,
    }


def validate_resume_config(
    existing_config: dict[str, Any],
    new_config: dict[str, Any],
):
    # Количество потоков можно менять при продолжении.
    old = dict(existing_config)
    new = dict(new_config)

    old.pop("workers", None)
    new.pop("workers", None)

    if old != new:
        raise ValueError(
            "Параметры нового запуска отличаются от исходного. "
            "Нельзя продолжить эксперимент с другой конфигурацией.\n"
            f"Существующие параметры: {old}\n"
            f"Новые параметры: {new}"
        )


def run_llm_parallel(
    run_id: str,
    pool_number: int,
    dataset: list[dict[str, Any]],
    evaluator: DatasetEvaluator,
    tracker: ExperimentTracker,
    workers: int,
    max_inflight: int,
):
    pending_indices = [
        index
        for index in range(len(dataset))
        if not tracker.is_record_completed(
            run_id,
            pool_number,
            index,
        )
    ]

    logger.info(
        "Pool %s: всего %s, готово %s, осталось %s.",
        pool_number,
        len(dataset),
        len(dataset) - len(pending_indices),
        len(pending_indices),
    )

    def process(index: int):
        return evaluator.evaluate_record(
            dataset[index],
            index,
        )

    iterator = iter(pending_indices)
    futures = {}

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=f"llm-pool-{pool_number}",
    ) as executor:

        with tqdm(
            total=len(pending_indices),
            desc=f"LLM pool {pool_number}",
        ) as progress:

            def submit_next():
                try:
                    index = next(iterator)
                except StopIteration:
                    return False

                future = executor.submit(process, index)
                futures[future] = index

                return True

            for _ in range(min(max_inflight, len(pending_indices))):
                submit_next()

            while futures:
                completed, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in completed:
                    index = futures.pop(future)

                    try:
                        result = future.result()

                        tracker.save_record(
                            run_id=run_id,
                            pool=pool_number,
                            record_index=index,
                            status="completed",
                            result=result,
                        )

                    except Exception as exc:
                        logger.exception(
                            "Ошибка: pool=%s, record=%s",
                            pool_number,
                            index,
                        )

                        tracker.save_record(
                            run_id=run_id,
                            pool=pool_number,
                            record_index=index,
                            status="error",
                            error=str(exc),
                        )

                    progress.update(1)

                    submit_next()


def calculate_mentions(
    run_id: str,
    pool_number: int,
    tracker: ExperimentTracker,
    embedder: MentionEmbedder,
):
    """
    Централизованное вычисление embeddings.

    LLM-ответы уже сохранены в SQLite.
    Здесь вычисляем similarity и обновляем записи.
    """

    records = tracker.load_records(
        run_id=run_id,
        pool=pool_number,
    )

    texts = []
    tasks = []

    for record_position, record in enumerate(records):
        gold = record["gold"]
        predictions = record["predictions"]

        for criterion_position, (gold_item, prediction) in enumerate(
            zip(gold, predictions)
        ):
            if str(gold_item.get("answer", "0")) != "1":
                prediction["mention_similarity"] = 0.0
                prediction["mention_exact_match"] = False
                prediction["mention_match"] = False
                continue

            gold_mention = str(
                gold_item.get("mention", "")
            ).strip()

            predicted_mention = str(
                prediction.get("mention", "")
            ).strip()

            prediction["mention_exact_match"] = (
                bool(predicted_mention)
                and gold_mention == predicted_mention
            )

            if not gold_mention or not predicted_mention:
                prediction["mention_similarity"] = 0.0
                prediction["mention_match"] = False
                continue

            texts.extend([
                gold_mention,
                predicted_mention,
            ])

            tasks.append((
                record_position,
                criterion_position,
            ))

    logger.info(
        "Pool %s: подготовлено %s текстов для embeddings.",
        pool_number,
        len(texts),
    )

    if texts:
        vectors = embedder.encode(texts)

        for task_position, (
            record_position,
            criterion_position,
        ) in enumerate(tasks):

            gold_vector = vectors[task_position * 2]
            predicted_vector = vectors[task_position * 2 + 1]

            similarity = float(
                (gold_vector * predicted_vector).sum()
            )

            prediction = records[record_position]["predictions"][
                criterion_position
            ]

            prediction["mention_similarity"] = similarity

            prediction["mention_match"] = (
                similarity >= embedder.mention_threshold
                if hasattr(embedder, "mention_threshold")
                else False
            )

    # Сохраняем обновлённые результаты.
    # Индексы записей восстановим через record_index.
    for record in records:
        tracker.save_record(
            run_id=run_id,
            pool=pool_number,
            record_index=record["record_index"],
            status="completed",
            result=record,
        )

    return records


def evaluate_pool(
    run_id: str,
    pool_number: int,
    args,
    evaluator: DatasetEvaluator,
    embedder: MentionEmbedder,
    tracker: ExperimentTracker,
):
    input_path = DATA_DIR / f"pool_{pool_number}_prepared.json"

    logger.info("Загрузка %s", input_path)

    dataset = load_dataset(input_path)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    max_inflight = args.max_inflight or args.workers * 2

    run_llm_parallel(
        run_id=run_id,
        pool_number=pool_number,
        dataset=dataset,
        evaluator=evaluator,
        tracker=tracker,
        workers=args.workers,
        max_inflight=max_inflight,
    )

    # Ошибочные записи не включаем в метрики качества.
    # Они учитываются отдельно как ошибки запуска.
    records = calculate_mentions(
        run_id=run_id,
        pool_number=pool_number,
        tracker=tracker,
        embedder=embedder,
    )

    overall, criterion_metrics = calculate_metrics(
        records=records,
        mention_threshold=args.mention_threshold,
    )

    statuses = tracker.get_record_status_counts(
        run_id,
        pool_number,
    )

    overall["failed_records"] = statuses.get("error", 0)
    overall["completed_records"] = statuses.get("completed", 0)
    overall["expected_records"] = len(dataset)

    tracker.save_metrics(
        run_id=run_id,
        pool=pool_number,
        overall=overall,
        criterion_metrics=criterion_metrics,
    )

    run_dir = args.output_dir / "runs" / run_id
    pool_dir = run_dir / f"pool_{pool_number}"

    save_json(
        pool_dir / "results.json",
        records,
    )

    save_json(
        pool_dir / "metrics.json",
        overall,
    )

    save_json(
        pool_dir / "criterion_metrics.json",
        criterion_metrics,
    )

    logger.info(
        "Pool %s | accuracy=%.4f | F1=%.4f | "
        "mention similarity=%.4f | errors=%s",
        pool_number,
        overall["accuracy"],
        overall["f1"],
        overall["mention_similarity_mean"],
        statuses.get("error", 0),
    )


def main():
    setup_logging()
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers должен быть не меньше 1.")

    if args.max_inflight is not None and args.max_inflight < 1:
        raise ValueError("--max-inflight должен быть не меньше 1.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit должен быть не меньше 1.")

    if not 0.0 <= args.mention_threshold <= 1.0:
        raise ValueError("--mention-threshold должен быть от 0 до 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(args.database)

    dataset_hashes = {}

    for pool_number in args.pools:
        path = DATA_DIR / f"pool_{pool_number}_prepared.json"

        if not path.exists():
            raise FileNotFoundError(path)

        dataset_hashes[str(pool_number)] = sha256_file(path)

    config = build_run_config(
        args,
        dataset_hashes,
    )

    if args.resume_run_id:
        run_id = args.resume_run_id
        existing = tracker.get_run(run_id)

        if not existing:
            raise ValueError(
                f"Эксперимент не найден: {run_id}"
            )

        existing_config = json.loads(
            existing["config_json"]
        )

        validate_resume_config(
            existing_config,
            config,
        )

        logger.info("Продолжаем эксперимент: %s", run_id)

    else:
        run_id = args.run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )

        tracker.create_run(
            run_id=run_id,
            experiment_name=args.experiment_name,
            provider=args.provider,
            model_name=args.model_name,
            config=config,
        )

    logger.info("Run ID: %s", run_id)
    logger.info("Provider: %s", args.provider)
    logger.info("Model: %s", args.model_name)
    logger.info("Workers: %s", args.workers)

    try:
        llm = LLMClient(
            provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        evaluator = DatasetEvaluator(llm)

        embedder = MentionEmbedder(
            model_name=args.embedding_model,
            device=args.device,
            batch_size=args.embedding_batch_size,
        )

        # Порог используется при расчёте similarity.
        embedder.mention_threshold = args.mention_threshold

        for pool_number in args.pools:
            evaluate_pool(
                run_id=run_id,
                pool_number=pool_number,
                args=args,
                evaluator=evaluator,
                embedder=embedder,
                tracker=tracker,
            )

        all_statuses = []

        for pool_number in args.pools:
            all_statuses.append(
                tracker.get_record_status_counts(
                    run_id,
                    pool_number,
                )
            )

        has_errors = any(
            statuses.get("error", 0) > 0
            for statuses in all_statuses
        )

        status = (
            "completed_with_errors"
            if has_errors
            else "completed"
        )

        tracker.update_run_status(run_id, status)

        exported = tracker.export_run(run_id)

        run_dir = args.output_dir / "runs" / run_id

        save_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "config": config,
                "status": status,
                "finished_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

        logger.info("Эксперимент завершён: %s", run_id)
        logger.info("Статус: %s", status)
        logger.info("SQLite: %s", args.database)
        logger.info("Результаты: %s", run_dir)

    except Exception:
        tracker.update_run_status(run_id, "failed")
        logger.exception("Эксперимент завершился с ошибкой.")
        raise


if __name__ == "__main__":
    main()



# python main.py \
#     --provider direct \
#     --model-name DeepSeek-V4.1-Flash \
#     --experiment-name baseline \
#     --pools 1 \
#     --limit 10 \
#     --workers 4 \
#     --max-inflight 8