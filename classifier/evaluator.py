from collections import defaultdict
from typing import Any

from classifier.llm import LLMClient
from prompts.sales_evaluation import SYSTEM_PROMPT


def validate_predictions(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    predictions_by_name = defaultdict(list)

    for prediction in predictions:
        predictions_by_name[prediction["name"]].append(
            prediction
        )

    validated = []

    for gold_item in gold:
        name = gold_item["name"]
        candidates = predictions_by_name[name]

        if candidates:
            prediction = candidates.pop(0)

            validated.append({
                "name": name,
                "answer": str(prediction["answer"]),
                "comment": str(prediction.get("comment", "")),
                "mention": str(prediction.get("mention", "")),
            })
        else:
            validated.append({
                "name": name,
                "answer": "0",
                "comment": "Модель не вернула результат по критерию.",
                "mention": "",
            })

    extras = []

    for remaining in predictions_by_name.values():
        extras.extend(remaining)

    return validated, extras


class DatasetEvaluator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate_record(
        self,
        record: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:

        gold = record["result"]

        predictions, metadata = self.llm.evaluate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=record["part_system_prompt"],
        )

        validated, extras = validate_predictions(
            gold=gold,
            predictions=predictions,
        )

        return {
            "record_index": index,
            "gold": gold,
            "predictions": validated,
            "extra_predictions": extras,
            "llm_metadata": metadata,
        }