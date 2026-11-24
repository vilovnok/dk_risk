
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

from classifier.evaluator import DatasetEvaluator
from classifier.llm import LLMClient
from classifier.mention_checker import MentionChecker
from classifier.metrics import calculate_metrics
from classifier.tracker import ExperimentTracker

from config import (
    DATA_DIR,
    DATABASE_PATH,
    DEFAULT_MODEL,
    DEFAULT_WORKERS,
    MAX_TOKENS,
    OUTPUT_DIR,
    RUNS_DIR,
    TEMPERATURE,
    TOP_P,
)

from prompts.sales_evaluation import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Многопоточная оценка моделей для проекта продаж ДК."
        )
    )

    # Основная LLM
    parser.add_argument(
        "--provider",
        choices=["aihf", "direct"],
        required=True,
        help="Провайдер основной модели.",
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Название основной модели.",
    )

    # LLM для проверки mention
    parser.add_argument(
        "--mention-provider",
        choices=["aihf", "direct"],
        default=None,
        help="Провайдер модели проверки mention. По умолчанию основной.",
    )

    parser.add_argument(
        "--mention-model-name",
        default=None,
        help="Модель проверки mention. По умолчанию основная.",
    )

    parser.add_argument(
        "--mention-max-tokens",
        type=int,
        default=4000,
        help="Максимальное количество токенов для проверки mention.",
    )

    # Experiment
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

    # Parallelism
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

    # Main LLM parameters
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

    # Dataset
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    # Run management
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

    # Paths
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


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

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

        if not isinstance(
            record.get("part_system_prompt"),
            str,
        ):
            raise ValueError(
                f"Некорректный part_system_prompt: запись {index}"
            )

        if not isinstance(record.get("result"), list):
            raise ValueError(
                f"Некорректный result: запись {index}"
            )

        for criterion_index, criterion in enumerate(
            record["result"]
        ):
            if not isinstance(criterion, dict):
                raise ValueError(
                    f"Критерий {criterion_index} записи {index} "
                    "должен быть объектом."
                )

            required_fields = {
                "name",
                "answer",
                "comment",
                "mention",
            }

            missing = required_fields - criterion.keys()

            if missing:
                raise ValueError(
                    f"В критерии {criterion_index} записи {index} "
                    f"отсутствуют поля: {missing}"
                )

            if str(criterion["answer"]) not in {"0", "1"}:
                raise ValueError(
                    f"Некорректный answer в записи {index}: "
                    f"{criterion['answer']}"
                )

    return data


# ---------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------

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

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

def build_run_config(args, dataset_hashes):
    prompt_hash = hashlib.sha256(
        SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()

    mention_provider = (
        args.mention_provider or args.provider
    )

    mention_model_name = (
        args.mention_model_name or args.model_name
    )

    return {
        "provider": args.provider,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,

        "mention_provider": mention_provider,
        "mention_model_name": mention_model_name,
        "mention_temperature": 0.0,
        "mention_top_p": 1.0,
        "mention_max_tokens": args.mention_max_tokens,

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
    # Количество потоков можно менять.
    # Остальные параметры моделей и датасета должны совпадать.
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


# ---------------------------------------------------------------------
# Parallel LLM evaluation
# ---------------------------------------------------------------------

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

    completed_count = (
        len(dataset) - len(pending_indices)
    )

    logger.info(
        "Pool %s: всего %s, готово %s, осталось %s.",
        pool_number,
        len(dataset),
        completed_count,
        len(pending_indices),
    )

    if not pending_indices:
        logger.info(
            "Pool %s: все записи уже обработаны.",
            pool_number,
        )
        return

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
            desc=f"Pool {pool_number}",
        ) as progress:

            def submit_next():
                try:
                    index = next(iterator)
                except StopIteration:
                    return False

                future = executor.submit(
                    process,
                    index,
                )

                futures[future] = index

                return True

            for _ in range(
                min(max_inflight, len(pending_indices))
            ):
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
                            "Ошибка обработки: pool=%s, record=%s",
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


# ---------------------------------------------------------------------
# Pool evaluation
# ---------------------------------------------------------------------

def evaluate_pool(
    run_id: str,
    pool_number: int,
    args,
    evaluator: DatasetEvaluator,
    tracker: ExperimentTracker,
):
    input_path = (
        DATA_DIR / f"pool_{pool_number}_prepared.json"
    )

    logger.info(
        "Загрузка датасета: %s",
        input_path,
    )

    dataset = load_dataset(input_path)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    max_inflight = (
        args.max_inflight
        if args.max_inflight is not None
        else args.workers * 2
    )

    # Этап 1:
    # Основная LLM оценивает критерии и формирует mention.
    #
    # Этап 2:
    # MentionChecker вызывает проверяющую LLM
    # и сравнивает эталонную и предсказанную цитаты.
    #
    # Оба этапа выполняются внутри evaluator.evaluate_record().
    run_llm_parallel(
        run_id=run_id,
        pool_number=pool_number,
        dataset=dataset,
        evaluator=evaluator,
        tracker=tracker,
        workers=args.workers,
        max_inflight=max_inflight,
    )

    # Загружаем сохранённые результаты из SQLite.
    records = tracker.load_records(
        run_id=run_id,
        pool=pool_number,
    )

    # Метрики классификации и LLM-проверки mention.
    overall, criterion_metrics = calculate_metrics(
        records=records,
    )

    statuses = tracker.get_record_status_counts(
        run_id,
        pool_number,
    )

    overall["failed_records"] = statuses.get(
        "error",
        0,
    )

    overall["completed_records"] = statuses.get(
        "completed",
        0,
    )

    overall["expected_records"] = len(dataset)

    tracker.save_metrics(
        run_id=run_id,
        pool=pool_number,
        overall=overall,
        criterion_metrics=criterion_metrics,
    )

    # Экспорт результатов.
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
        "mention semantic=%.4f | errors=%s",
        pool_number,
        overall.get("accuracy", 0.0),
        overall.get("f1", 0.0),
        overall.get("mention_semantic_accuracy", 0.0),
        statuses.get("error", 0),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    setup_logging()

    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers должен быть не меньше 1."
        )

    if (
        args.max_inflight is not None
        and args.max_inflight < 1
    ):
        raise ValueError(
            "--max-inflight должен быть не меньше 1."
        )

    if args.limit is not None and args.limit < 1:
        raise ValueError(
            "--limit должен быть не меньше 1."
        )

    if args.max_tokens < 1:
        raise ValueError(
            "--max-tokens должен быть больше 0."
        )

    if args.mention_max_tokens < 1:
        raise ValueError(
            "--mention-max-tokens должен быть больше 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Tracker
    # -------------------------------------------------------------

    tracker = ExperimentTracker(
        args.database,
    )

    # -------------------------------------------------------------
    # Dataset hashes
    # -------------------------------------------------------------

    dataset_hashes = {}

    for pool_number in args.pools:
        path = (
            DATA_DIR
            / f"pool_{pool_number}_prepared.json"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        dataset_hashes[str(pool_number)] = (
            sha256_file(path)
        )

    # -------------------------------------------------------------
    # Experiment configuration
    # -------------------------------------------------------------

    config = build_run_config(
        args,
        dataset_hashes,
    )

    if args.resume_run_id:
        run_id = args.resume_run_id

        existing = tracker.get_run(
            run_id,
        )

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

        logger.info(
            "Продолжаем эксперимент: %s",
            run_id,
        )

    else:
        run_id = (
            args.run_id
            or (
                datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                + "_"
                + uuid.uuid4().hex[:8]
            )
        )

        tracker.create_run(
            run_id=run_id,
            experiment_name=args.experiment_name,
            provider=args.provider,
            model_name=args.model_name,
            config=config,
        )

    # -------------------------------------------------------------
    # LLM clients
    # -------------------------------------------------------------

    mention_provider = (
        args.mention_provider or args.provider
    )

    mention_model_name = (
        args.mention_model_name or args.model_name
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Основная LLM: provider=%s, model=%s",
        args.provider,
        args.model_name,
    )

    logger.info(
        "Mention LLM: provider=%s, model=%s",
        mention_provider,
        mention_model_name,
    )

    logger.info(
        "Workers: %s",
        args.workers,
    )

    # Основной клиент.
    llm_client = LLMClient(
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Отдельный клиент для проверки mention.
    mention_client = LLMClient(
        provider=mention_provider,
        model_name=mention_model_name,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.mention_max_tokens,
    )

    mention_checker = MentionChecker(
        mention_client,
    )

    evaluator = DatasetEvaluator(
        llm_client=llm_client,
        mention_checker=mention_checker,
    )

    # -------------------------------------------------------------
    # Evaluate pools
    # -------------------------------------------------------------

    try:
        for pool_number in args.pools:
            evaluate_pool(
                run_id=run_id,
                pool_number=pool_number,
                args=args,
                evaluator=evaluator,
                tracker=tracker,
            )

        # ---------------------------------------------------------
        # Final status
        # ---------------------------------------------------------

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

        tracker.update_run_status(
            run_id,
            status,
        )

        # Экспорт из SQLite.
        tracker.export_run(
            run_id,
        )

        run_dir = (
            args.output_dir
            / "runs"
            / run_id
        )

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

        logger.info(
            "Эксперимент завершён: %s",
            run_id,
        )

        logger.info(
            "Статус: %s",
            status,
        )

        logger.info(
            "SQLite: %s",
            args.database,
        )

        logger.info(
            "Результаты: %s",
            run_dir,
        )

    except Exception:
        tracker.update_run_status(
            run_id,
            "failed",
        )

        logger.exception(
            "Эксперимент завершился с ошибкой."
        )

        raise


if __name__ == "__main__":
    main()


# python main.py \
#     --provider direct \
#     --model-name DeepSeek-V4.1-Flash \
#     --mention-provider aihf \
#     --mention-model-name DeepSeek-V4.1-Flash \
#     --experiment-name mention-validation \
#     --pools 1 \
#     --workers 4