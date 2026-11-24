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