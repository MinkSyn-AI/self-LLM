import os
from pathlib import Path
from typing import List, Type

from sentence_transformers import SentenceTransformer

from ..base import ModuleEngine
from .embed_base import EmbeddingType


class TransformerEmbedding(ModuleEngine):
    embed_type: EmbeddingType = EmbeddingType.CONFIG

    @classmethod
    def from_builded(
        cls, name: str = "AITeamVN/Vietnamese_Embedding", **kwargs
    ) -> Type[ModuleEngine]:
        try:
            if os.path.exists(name):
                cls.name = Path(name).name
            else:
                cls.name = name

            cls.embedding_model = SentenceTransformer(name, trust_remote_code=True)

        except Exception as e:
            raise ValueError(
                f"SentenceTransformer failed to initialize. Error: {e}"
            ) from e

        return cls()

    def execute(self, docs: List[str], **kwargs):
        try:
            return [self.embedding_model.encode(text) for text in docs]

        except Exception as e:
            raise ValueError(f"Failed to get embeddings. Error details: {e}") from e
