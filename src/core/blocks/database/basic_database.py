from loguru import logger

from core.blocks.base import BaseBlocker
from core.modules import (
    AutoEmbdeddingModule,
    AutoGeneratorModule,
    AutoVectorSearchModule,
    BaseModule,
)


class RAGBasicBlocker(BaseBlocker):
    def __init__(
        self,
        generator: AutoGeneratorModule,
        retriever: AutoVectorSearchModule,
        embedder: AutoEmbdeddingModule,
    ):
        self.generator = generator
        self.retriever = retriever
        self.embedder = embedder

        self.embed_healthy = self.is_healthy()

    def _execute_base(self, user_query: str, **kwargs):
        retrieved_docs = []

        if self.embed_healthy:
            retrieved_docs = self.search_embedding(user_query, **kwargs)

        response = self.generate_response(user_query, retrieved_docs, **kwargs)
        return response

    def search_embedding(self, user_query: str, **kwargs) -> list[str]:
        query_embedding = self.get_embedding(user_query)
        return self.retriever.execute(
            query_embedding, kwargs.get('limit_embedding', 5), **kwargs
        )

    def generate_response(
        self, user_query: str, retrieved_docs: list[str] = [], **kwargs
    ) -> dict[str, any]:
        if not retrieved_docs:
            return self.generator.execute(user_query, **kwargs)

        context = "\n".join([doc for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer based on the context above:\n"

        return self.generator.execute(prompt=prompt, **kwargs)

    def get_embedding(self, text: str):
        if not text.strip():
            return []

        return self.embedder.execute(docs=[text])

    def is_healthy(self) -> bool:
        checks: dict[str, BaseModule] = {
            "Retriever": self.retriever,
            "Embedding": self.embedder,
            "LLM": self.generator,
        }

        embed_healthy = True

        for name, component in checks.items():
            if component.is_healthy():
                logger.success(f"{name} with {component.engine_name} engine exists.")
            elif name in ["Embedding", "Retriever"]:
                logger.warning(
                    f"{name} with {component.engine_name} engine does not exist."
                )
                embed_healthy = False
            else:
                logger.error(
                    f"{name} with {component.engine_name} engine does not exist."
                )
                raise ValueError(
                    "LLM generator is not healthy. Cannot generate response."
                )

        return embed_healthy
