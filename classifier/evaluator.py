from classifier.mention_checker import MentionChecker


class DatasetEvaluator:
    def __init__(
        self,
        llm_client,
        mention_checker: MentionChecker,
    ):
        self.llm_client = llm_client
        self.mention_checker = mention_checker