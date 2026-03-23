import os
from pathlib import Path
from typing import Any, Optional, Type

from qdrant_client import QdrantClient

from ..base import ModuleEngine


class QdrantVectorEngine(ModuleEngine):

    @classmethod
    def from_builded(
        cls,
        name: str = 'Alibaba-NLP/gte-multilingual-base',
        qdrant_api: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        limit: int = 12,
        **kwargs,
    ) -> Type[ModuleEngine]:
        try:
            if os.path.exists(name):
                cls.name = Path(name).name
            else:
                cls.name = name
            cls.limit = limit

            cls.collection = cls.name
            cls.client = QdrantClient(url=qdrant_url, api_key=qdrant_api)

        except Exception as e:
            raise ValueError(f"Qdrant failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, query_vector: Any, **kwargs):
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=kwargs.get('limit_embedding', self.limit),
            )

            results = []
            for hit in hits:
                results.append(
                    {
                        '_id': hit.payload['_id'],
                        'combined_information': hit.payload['combined_information'],
                        'score': hit.score,
                    }
                )
            return results

        except Exception as e:
            raise ValueError(f"Failed to query embeddings. Error details: {e}") from e
