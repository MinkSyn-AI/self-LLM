from typing import List, Type

from fastembed import TextEmbedding

from ..base import ModuleEngine
from .embed_base import EmbeddingType


class FastEmbeddingEngine(ModuleEngine):
    embed_type: EmbeddingType = EmbeddingType.LOCAL

    @classmethod
    def from_builded(
        cls, name: str = 'BAAI/bge-m3', max_length: int = 512, **kwargs
    ) -> Type[ModuleEngine]:
        try:
            cls.embedding_model = TextEmbedding(name=name, max_length=max_length)

        except Exception as e:
            raise ValueError(f"Fastembed failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, docs: List[str], **kwargs):
        try:
            embeds = self.embedding_model.embed(docs)
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings

        except Exception as e:
            raise ValueError(f"Failed to get embeddings. Error details: {e}") from e
