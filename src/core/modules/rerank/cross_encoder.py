from typing import Type

from sentence_transformers import CrossEncoder

from ..base import ModuleEngine


class CrossEncoderRerankEngine(ModuleEngine):

    @classmethod
    def from_builded(
        cls, name: str = "Alibaba-NLP/gte-multilingual-reranker-base", **kwargs
    ) -> Type[ModuleEngine]:
        try:
            cls.name = name
            cls.reranker = CrossEncoder(name, trust_remote_code=True)

        except Exception as e:
            raise ValueError(f"CrossEncoder failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, query: str, passages: list[str], **kwargs):
        if not passages:
            return [], []

        query_passage_pairs = [[query, passage] for passage in passages]

        scores = self.reranker.predict(query_passage_pairs)
        scored_passages = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

        ranked_scores = [float(score) for score, _ in scored_passages]
        ranked_passages = [passage for _, passage in scored_passages]

        return ranked_scores, ranked_passages
