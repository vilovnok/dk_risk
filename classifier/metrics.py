from collections import defaultdict
from typing import Any

import numpy as np


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def binary_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, Any]:

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    total = tp + fp + fn + tn

    return {
        "accuracy": safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": total,
    }


def calculate_metrics(
    records: list[dict[str, Any]],
    mention_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    counters = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "mention_total": 0,
            "mention_exact": 0,
            "mention_semantic": 0,
            "similarities": [],
        }
    )

    global_counter = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "mention_total": 0,
        "mention_exact": 0,
        "mention_semantic": 0,
        "similarities": [],
    }

    total_criteria = 0
    extra_predictions = 0

    for record in records:
        gold = record.get("gold", [])
        predictions = record.get("predictions", [])

        extra_predictions += len(
            record.get("extra_predictions", [])
        )

        for gold_item, prediction in zip(gold, predictions):
            name = gold_item["name"]

            gold_answer = str(gold_item.get("answer", "0"))
            pred_answer = str(prediction.get("answer", "0"))

            gold_positive = gold_answer == "1"
            pred_positive = pred_answer == "1"

            if gold_positive and pred_positive:
                label = "tp"
            elif not gold_positive and pred_positive:
                label = "fp"
            elif gold_positive and not pred_positive:
                label = "fn"
            else:
                label = "tn"

            total_criteria += 1

            for counter in (global_counter, counters[name]):
                counter[label] += 1

            if not gold_positive:
                continue

            gold_mention = normalize_text(
                gold_item.get("mention", "")
            )

            predicted_mention = normalize_text(
                prediction.get("mention", "")
            )

            exact = (
                bool(predicted_mention)
                and gold_mention == predicted_mention
            )

            similarity = float(
                prediction.get("mention_similarity", 0.0)
            )

            semantic = (
                bool(predicted_mention)
                and similarity >= mention_threshold
            )

            for counter in (global_counter, counters[name]):
                counter["mention_total"] += 1
                counter["mention_exact"] += int(exact)
                counter["mention_semantic"] += int(semantic)

                counter["similarities"].append(similarity)

    def finalize(counter: dict[str, Any]) -> dict[str, Any]:
        result = binary_metrics(
            tp=counter["tp"],
            fp=counter["fp"],
            fn=counter["fn"],
            tn=counter["tn"],
        )

        mention_total = counter["mention_total"]
        similarities = counter["similarities"]

        result.update({
            "mention_support": mention_total,
            "mention_exact_accuracy": safe_div(
                counter["mention_exact"],
                mention_total,
            ),
            "mention_semantic_accuracy": safe_div(
                counter["mention_semantic"],
                mention_total,
            ),
            "mention_similarity_mean": (
                float(np.mean(similarities))
                if similarities else 0.0
            ),
            "mention_threshold": mention_threshold,
        })

        return result

    overall = finalize(global_counter)

    overall["total_criteria"] = total_criteria
    overall["extra_predictions"] = extra_predictions
    overall["records_count"] = len(records)

    criterion_metrics = []

    for name, counter in sorted(counters.items()):
        criterion_metrics.append({
            "criterion_name": name,
            "metrics": finalize(counter),
        })

    return overall, criterion_metrics



def calculate_mention_metrics(records):
    checks = [
        check
        for record in records
        if record.get("status") == "completed"
        for check in record.get("mention_checks", [])
    ]

    total = len(checks)

    if total == 0:
        return {
            "mention_total": 0,
            "mention_exact_accuracy": 0.0,
            "mention_semantic_accuracy": 0.0,
            "mention_grounded_accuracy": 0.0,
            "mention_completeness_accuracy": 0.0,
        }

    return {
        "mention_total": total,
        "mention_exact_accuracy": sum(
            x["exact_match"] for x in checks
        ) / total,
        "mention_semantic_accuracy": sum(
            x["semantic_match"] for x in checks
        ) / total,
        "mention_grounded_accuracy": sum(
            x["grounded"] for x in checks
        ) / total,
        "mention_completeness_accuracy": sum(
            x["complete"] for x in checks
        ) / total,
    }



def calculate_mention_metrics(records):
    checks = [
        check
        for record in records
        if record.get("status") == "completed"
        for check in record.get("mention_checks", [])
    ]

    total = len(checks)

    if total == 0:
        return {
            "mention_total": 0,
            "mention_exact_accuracy": 0.0,
            "mention_semantic_accuracy": 0.0,
            "mention_grounded_accuracy": 0.0,
            "mention_completeness_accuracy": 0.0,
            "mention_missing_rate": 0.0,
        }

    return {
        "mention_total": total,

        "mention_exact_accuracy": sum(
            bool(x.get("exact_match", False))
            for x in checks
        ) / total,

        "mention_semantic_accuracy": sum(
            bool(x.get("semantic_match", False))
            for x in checks
        ) / total,

        "mention_grounded_accuracy": sum(
            bool(x.get("grounded", False))
            for x in checks
        ) / total,

        "mention_completeness_accuracy": sum(
            bool(x.get("complete", False))
            for x in checks
        ) / total,

        "mention_missing_rate": sum(
            not bool(x.get("predicted_mention"))
            for x in checks
        ) / total,
    }
