from core.blocks.rag import RAGBasicBlocker, RAGRerankBlocker
from core.modules import (
    AutoEmbdeddingModule,
    AutoGeneratorModule,
    AutoRerankModule,
    AutoVectorSearchModule,
    EmbeddingName,
    GeneratorName,
    RerankName,
    VectorSearchName,
)


def _build_modules():
    # Default configurations for modules
    embedder = AutoEmbdeddingModule.from_builded(engine_name=EmbeddingName.Transformers)

    # MongoDB Vector Search Configuration
    mongo_config = {
        "name": "vcm-mongodb",
        "mongodbUri": "mongodb://ftecher:ftech132@160.30.129.168:27015/?authSource=admin",
        "mongoCollection": "product_embeddings",
        "limit": 20,
    }
    retriever = AutoVectorSearchModule.from_builded(
        engine_name=VectorSearchName.MongoDB, **mongo_config
    )

    # Gemini Generator Configuration
    gemini_config = {
        "name": "gemini-2.5-flash",
        "api_key": "AIzaSyDogxB7Fr4zsbRWyohAyOnMK7oxqeWBSoM",
    }
    generator = AutoGeneratorModule.from_builded(
        engine_name=GeneratorName.Gemini, **gemini_config
    )

    # SentenceTransformer Reranker Configuration
    reranker = AutoRerankModule.from_builded(engine_name=RerankName.CrossEncoder)

    return embedder, retriever, generator, reranker


def execute_basic_rag_example(
    generator: AutoGeneratorModule,
    retriever: AutoVectorSearchModule,
    embedder: AutoEmbdeddingModule,
):
    # Example usage of RAGBasicBlocker
    rag_blocker = RAGBasicBlocker(
        generator=generator, retriever=retriever, embedder=embedder
    )

    user_query = "What is the capital of France?"
    response = rag_blocker.execute(user_query=user_query)
    print(response)


def execute_rerank_rag_example(
    generator: AutoGeneratorModule,
    retriever: AutoVectorSearchModule,
    embedder: AutoEmbdeddingModule,
    reranker: AutoRerankModule,
):
    # Example usage of RAGBasicBlocker
    rag_blocker = RAGBasicBlocker(
        generator=generator, retriever=retriever, embedder=embedder, reranker=reranker
    )

    user_query = "What is the capital of France?"
    response = rag_blocker.execute(user_query)
    print(response)


if __name__ == "__main__":
    # Embedding using with Transformers engine -> EmbeddingName.Transformers
    # Retrieve using with MongoDB engine -> VectorSearchName.MongoDB
    # Generate using with Gemini engine -> GeneratorName.Gemini
    # Rerank using with SentenceTransformer engine -> RerankName.CrossEncoder
    embedder, retriever, generator, reranker = _build_modules()

    print("Executing RAG Basic Example:")
    execute_basic_rag_example(
        generator=generator, retriever=retriever, embedder=embedder
    )

    print("\nExecuting RAG Rerank Example:")
    execute_rerank_rag_example(
        generator=generator,
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
    )
