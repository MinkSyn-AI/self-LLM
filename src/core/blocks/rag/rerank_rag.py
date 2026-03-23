from core.blocks.rag.basic_rag import RAGBasicBlocker
from core.modules import (
    AutoEmbdeddingModule,
    AutoGeneratorModule,
    AutoRerankModule,
    AutoVectorSearchModule,
)


class RAGRerankBlocker(RAGBasicBlocker):
    def __init__(
        self,
        generator: AutoGeneratorModule,
        retriever: AutoVectorSearchModule,
        embedder: AutoEmbdeddingModule,
        reranker: AutoRerankModule,
    ):
        self.generator = generator
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker

        self.embed_healthy = self.is_healthy()

    def _execute_base(self, user_query: str, **kwargs):
        retrieved_docs = []

        if self.embed_healthy:
            retrieved_docs = self.search_embedding(user_query, **kwargs)

        if self.reranker.is_healthy() and retrieved_docs:
            doc_texts = [
                doc.get("combined_information", str(doc))
                if isinstance(doc, dict)
                else str(doc)
                for doc in retrieved_docs
            ]
            _, ranked_passages = self.reranker.execute(user_query, doc_texts, **kwargs)
            retrieved_docs = ranked_passages[: kwargs.get('limit_rerank', 5)]
        elif len(retrieved_docs) > kwargs.get('limit_rerank', 5):
            retrieved_docs = retrieved_docs[: kwargs.get('limit_rerank', 5)]

        response = self.generate_response(user_query, retrieved_docs, **kwargs)
        return response
