import json
import sqlite3

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentTracker:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=60,
        )

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS record_results (
                    run_id TEXT NOT NULL,
                    pool INTEGER NOT NULL,
                    record_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,

                    PRIMARY KEY (
                        run_id,
                        pool,
                        record_index
                    ),

                    FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS pool_metrics (
                    run_id TEXT NOT NULL,
                    pool INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,

                    PRIMARY KEY (run_id, pool),

                    FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS criterion_metrics (
                    run_id TEXT NOT NULL,
                    pool INTEGER NOT NULL,
                    criterion_name TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,

                    PRIMARY KEY (
                        run_id,
                        pool,
                        criterion_name
                    ),

                    FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
                );
            """)

    def create_run(
        self,
        run_id: str,
        experiment_name: str,
        provider: str,
        model_name: str,
        config: dict[str, Any],
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    experiment_name,
                    provider,
                    model_name,
                    config_json,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_name,
                    provider,
                    model_name,
                    json.dumps(config, ensure_ascii=False),
                    "running",
                    now_iso(),
                ),
            )

    def get_run(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        return dict(row) if row else None

    def update_run_status(
        self,
        run_id: str,
        status: str,
    ):
        finished_at = (
            now_iso()
            if status in {
                "completed",
                "completed_with_errors",
                "failed",
            }
            else None
        )

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (status, finished_at, run_id),
            )

    def is_record_completed(
        self,
        run_id: str,
        pool: int,
        record_index: int,
    ) -> bool:

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status
                FROM record_results
                WHERE run_id = ?
                  AND pool = ?
                  AND record_index = ?
                """,
                (run_id, pool, record_index),
            ).fetchone()

        return bool(
            row and row["status"] == "completed"
        )

    def save_record(
        self,
        run_id: str,
        pool: int,
        record_index: int,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        result_json = (
            json.dumps(result, ensure_ascii=False)
            if result is not None
            else None
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO record_results (
                    run_id,
                    pool,
                    record_index,
                    status,
                    result_json,
                    error,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT (
                    run_id,
                    pool,
                    record_index
                )
                DO UPDATE SET
                    status = excluded.status,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    pool,
                    record_index,
                    status,
                    result_json,
                    error,
                    now_iso(),
                ),
            )

    def load_records(
        self,
        run_id: str,
        pool: int,
        status: str = "completed",
    ) -> list[dict[str, Any]]:

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT result_json
                FROM record_results
                WHERE run_id = ?
                  AND pool = ?
                  AND status = ?
                ORDER BY record_index
                """,
                (run_id, pool, status),
            ).fetchall()

        return [
            json.loads(row["result_json"])
            for row in rows
            if row["result_json"]
        ]

    def get_record_status_counts(
        self,
        run_id: str,
        pool: int,
    ) -> dict[str, int]:

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM record_results
                WHERE run_id = ?
                  AND pool = ?
                GROUP BY status
                """,
                (run_id, pool),
            ).fetchall()

        return {
            row["status"]: row["count"]
            for row in rows
        }

    def save_metrics(
        self,
        run_id: str,
        pool: int,
        overall: dict[str, Any],
        criterion_metrics: list[dict[str, Any]],
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pool_metrics (
                    run_id,
                    pool,
                    metrics_json
                )
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    pool,
                    json.dumps(overall, ensure_ascii=False),
                ),
            )

            conn.execute(
                """
                DELETE FROM criterion_metrics
                WHERE run_id = ? AND pool = ?
                """,
                (run_id, pool),
            )

            for item in criterion_metrics:
                conn.execute(
                    """
                    INSERT INTO criterion_metrics (
                        run_id,
                        pool,
                        criterion_name,
                        metrics_json
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        pool,
                        item["criterion_name"],
                        json.dumps(
                            item["metrics"],
                            ensure_ascii=False,
                        ),
                    ),
                )

    def export_run(
        self,
        run_id: str,
    ) -> dict[str, Any]:

        with self._connect() as conn:
            run = conn.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if not run:
                raise ValueError(
                    f"Run not found: {run_id}"
                )

            records = conn.execute(
                """
                SELECT *
                FROM record_results
                WHERE run_id = ?
                ORDER BY pool, record_index
                """,
                (run_id,),
            ).fetchall()

            pool_metrics = conn.execute(
                """
                SELECT *
                FROM pool_metrics
                WHERE run_id = ?
                ORDER BY pool
                """,
                (run_id,),
            ).fetchall()

            criterion_metrics = conn.execute(
                """
                SELECT *
                FROM criterion_metrics
                WHERE run_id = ?
                ORDER BY pool, criterion_name
                """,
                (run_id,),
            ).fetchall()

        return {
            "run": dict(run),
            "records": [dict(row) for row in records],
            "pool_metrics": [
                dict(row) for row in pool_metrics
            ],
            "criterion_metrics": [
                dict(row) for row in criterion_metrics
            ],
        }