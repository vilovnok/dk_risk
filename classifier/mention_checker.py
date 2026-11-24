
import json
import logging
from typing import Any

from classifier.llm import LLMClient

logger = logging.getLogger(__name__)


MENTION_CHECK_SYSTEM_PROMPT = """
Ты — независимый эксперт по проверке цитат в диалогах.

Твоя задача — сравнить эталонную цитату из датасета с цитатой,
сформированной другой языковой моделью.

Для каждого критерия оцени четыре независимых показателя:

1. exact_match:
   Дословно ли предсказанная цитата совпадает с эталонной.
   Допускается игнорировать различия в пробелах и переносах строк.
   Не считай перефразирование дословным совпадением.

2. semantic_match:
   Передаёт ли предсказанная цитата тот же смысл, что и эталонная.
   Не требуй буквального совпадения.
   Учитывай, что формулировки могут отличаться, но описывать
   одно и то же действие или содержание.

3. grounded:
   Подтверждается ли предсказанная цитата исходным диалогом.
   Цитата должна действительно присутствовать в диалоге.
   Нельзя считать подтверждением информацию, которую сотрудник
   не произносил.
   Не путай реплики клиента и сотрудника.

4. complete:
   Достаточно ли предсказанной цитаты для подтверждения критерия.
   Цитата может быть дословной и присутствовать в диалоге,
   но быть неполной или не подтверждать требуемое действие.

Правила:
- Эталонная цитата — ориентир для сравнения, но не абсолютная истина.
- Не признавай цитату правильной только потому, что она похожа
  на эталонную.
- Не признавай цитату grounded, если её содержание отсутствует
  в исходном диалоге.
- Если предсказанная цитата пустая, все четыре показателя false.
- Если эталонная цитата пустая, semantic_match и exact_match false.
- Не додумывай отсутствующий контекст.
- Оценивай каждый критерий независимо.

Верни только JSON-массив без Markdown и дополнительного текста.

Для каждого объекта используй формат:
{
  "name": "название критерия",
  "answer": "1",
  "comment": "Краткое обоснование",
  "mention": "предсказанная цитата",
  "exact_match": true,
  "semantic_match": true,
  "grounded": true,
  "complete": true
}

Поля exact_match, semantic_match, grounded и complete
должны содержать именно JSON boolean: true или false.
"""


class MentionChecker:
    def __init__(self, client: LLMClient):
        self.client = client

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def check(
        self,
        dialogue: str,
        gold: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:

        prediction_map = {
            item["name"]: item
            for item in predictions
        }

        check_items = []

        for item in gold:
            if str(item.get("answer", "0")) != "1":
                continue

            predicted = prediction_map.get(item["name"], {})

            gold_mention = str(item.get("mention", "") or "")
            predicted_mention = str(
                predicted.get("mention", "") or ""
            )

            check_items.append({
                "name": item["name"],
                "gold_mention": gold_mention,
                "predicted_answer": str(
                    predicted.get("answer", "0")
                ),
                "predicted_mention": predicted_mention,
            })

        if not check_items:
            return [], {}

        user_prompt = f"""
Исходный диалог:
<dialogue>
{dialogue}
</dialogue>

Цитаты для проверки:
{json.dumps(check_items, ensure_ascii=False, indent=2)}

Проверь каждую цитату относительно исходного диалога
и эталонной цитаты.
Верни результат для каждого переданного критерия.
Сохрани исходные названия критериев.
"""

        results, metadata = self.client.evaluate(
            system_prompt=MENTION_CHECK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        expected_names = {item["name"] for item in check_items}
        result_map = {}

        for item in results:
            name = item["name"]

            if name not in expected_names:
                continue

            result_map[name] = {
                "name": name,
                "exact_match": self._to_bool(
                    item.get("exact_match", False)
                ),
                "semantic_match": self._to_bool(
                    item.get("semantic_match", False)
                ),
                "grounded": self._to_bool(
                    item.get("grounded", False)
                ),
                "complete": self._to_bool(
                    item.get("complete", False)
                ),
                "comment": str(item.get("comment", "")),
            }

        aligned = []

        for item in check_items:
            name = item["name"]

            result = result_map.get(name)

            if result is None:
                result = {
                    "name": name,
                    "exact_match": False,
                    "semantic_match": False,
                    "grounded": False,
                    "complete": False,
                    "comment": "Проверяющая модель не вернула результат.",
                }

            aligned.append(result)

        return aligned, metadata

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {
                "true", "1", "yes"
            }

        return False
