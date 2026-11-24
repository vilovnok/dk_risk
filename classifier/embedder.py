import logging

import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class MentionEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        max_length: int = EMBEDDING_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device == "auto":
            device = None

        logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.model.max_seq_length = max_length

        logger.info("Embedding model loaded.")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )