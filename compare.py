import argparse
import csv
import json
import sqlite3

from pathlib import Path
from typing import Any

from config import DATABASE_PATH, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сравнение результатов экспериментов."
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "comparisons",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Фильтр по названию эксперимента.",
    )

    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="Список конкретных run_id.",
    )

    return parser.parse_args()


def load_data(
    database: Path,
    experiment_name: str | None,
    run_ids: list[str] | None,
):
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row

    try:
        query = """
            SELECT
                r.run_id,
                r.experiment_name,
                r.provider,
                r.model_name,
                r.config_json,
                r.status,
                r.started_at,
                r.finished_at,
                p.pool,
                p.metrics_json
            FROM runs r
            JOIN pool_metrics p
                ON r.run_id = p.run_id
            WHERE 1 = 1
        """

        params = []

        if experiment_name:
            query += " AND r.experiment_name = ?"
            params.append(experiment_name)

        if run_ids:
            placeholders = ",".join(
                "?" for _ in run_ids
            )

            query += f" AND r.run_id IN ({placeholders})"
            params.extend(run_ids)

        query += " ORDER BY r.started_at, p.pool"

        pool_rows = conn.execute(
            query,
            params,
        ).fetchall()

        criterion_query = """
            SELECT
                r.run_id,
                r.experiment_name,
                r.provider,
                r.model_name,
                c.pool,
                c.criterion_name,
                c.metrics_json
            FROM runs r
            JOIN criterion_metrics c
                ON r.run_id = c.run_id
            WHERE 1 = 1
        """

        criterion_params = []

        if experiment_name:
            criterion_query += " AND r.experiment_name = ?"
            criterion_params.append(experiment_name)

        if run_ids:
            placeholders = ",".join(
                "?" for _ in run_ids
            )

            criterion_query += (
                f" AND r.run_id IN ({placeholders})"
            )

            criterion_params.extend(run_ids)

        criterion_query += (
            " ORDER BY r.started_at, c.pool, c.criterion_name"
        )

        criterion_rows = conn.execute(
            criterion_query,
            criterion_params,
        ).fetchall()

        return pool_rows, criterion_rows

    finally:
        conn.close()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(f"Нет данных для: {path}")
        return

    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    if not args.database.exists():
        raise FileNotFoundError(args.database)

    pool_rows, criterion_rows = load_data(
        database=args.database,
        experiment_name=args.experiment_name,
        run_ids=args.run_ids,
    )

    model_comparison = []

    for row in pool_rows:
        metrics = json.loads(row["metrics_json"])
        config = json.loads(row["config_json"])

        model_comparison.append({
            "run_id": row["run_id"],
            "experiment_name": row["experiment_name"],
            "provider": row["provider"],
            "model_name": row["model_name"],
            "pool": row["pool"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "max_tokens": config.get("max_tokens"),
            "embedding_model": config.get("embedding_model"),
            "mention_threshold": config.get("mention_threshold"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tn": metrics.get("tn"),
            "mention_exact_accuracy": metrics.get(
                "mention_exact_accuracy"
            ),
            "mention_semantic_accuracy": metrics.get(
                "mention_semantic_accuracy"
            ),
            "mention_similarity_mean": metrics.get(
                "mention_similarity_mean"
            ),
            "completed_records": metrics.get(
                "completed_records"
            ),
            "failed_records": metrics.get(
                "failed_records"
            ),
            "extra_predictions": metrics.get(
                "extra_predictions"
            ),
        })

    criterion_comparison = []

    for row in criterion_rows:
        metrics = json.loads(row["metrics_json"])

        criterion_comparison.append({
            "run_id": row["run_id"],
            "experiment_name": row["experiment_name"],
            "provider": row["provider"],
            "model_name": row["model_name"],
            "pool": row["pool"],
            "criterion_name": row["criterion_name"],
            **metrics,
        })

    write_csv(
        args.output_dir / "model_comparison.csv",
        model_comparison,
    )

    write_csv(
        args.output_dir / "criterion_comparison.csv",
        criterion_comparison,
    )

    print(f"Экспериментов в сравнении: {len(pool_rows)}")
    print(
        "Сравнение моделей:",
        args.output_dir / "model_comparison.csv",
    )
    print(
        "Сравнение критериев:",
        args.output_dir / "criterion_comparison.csv",
    )


if __name__ == "__main__":
    main()