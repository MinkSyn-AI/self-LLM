from typing import Any, Type

import chromadb

from ..base import ModuleEngine


class ChormaDBVectorEngine(ModuleEngine):

    @classmethod
    def from_builded(
        cls,
        name: str = 'Alibaba-NLP/gte-multilingual-base',
        path: str = "./chroma_db",
        limit: int = 12,
        **kwargs,
    ) -> Type[ModuleEngine]:
        try:
            cls.client = chromadb.PersistentClient(path=path)
            cls.collection = cls.client.get_collection(name=name)
            cls.limit = limit

        except Exception as e:
            raise ValueError(f"ChromaDB failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, query_vector: Any, **kwargs):
        try:
            hits = self.collection.query(
                query_embeddings=[query_vector],
                n_results=kwargs.get('limit_embedding', self.limit),
            )

            results = []
            for i in range(len(hits['ids'][0])):
                distance = hits['distances'][0][i]
                simlarity = 1 - distance

                result = {
                    "_id": hits['ids'][0][i],
                    "combined_information": hits['documents'][0][i],
                    "score": simlarity,
                }
                results.append(result)
            return results

        except Exception as e:
            raise ValueError(f"Failed to query embeddings. Error details: {e}") from e
