import numpy as np

from ..base import BaseBlocker


class SemanticRouter(BaseBlocker):
    def __init__(self, embedding, routes):
        self.routes = routes
        self.embedding = embedding

        self.routesEmbedding = {}
        for route in self.routes:
            self.routesEmbedding[route.name] = self._encode(route.samples)

    def get_routes(self):
        return self.routes

    def _execute_base(self, query, **kwargs):
        queryEmbedding = self._encode([query])
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        scores = []

        # Calculate the cosine similarity of the query embedding with the sample embeddings of the router.
        for route in self.routes:
            routesEmbedding = self.routesEmbedding[route.name] / np.linalg.norm(
                self.routesEmbedding[route.name]
            )
            score = np.mean(np.dot(routesEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]

    # Backward-compatible alias in case other code still calls the typo.
    def _excute_base(self, query, **kwargs):
        return self._execute_base(query=query, **kwargs)

    def _encode(self, docs):
        if hasattr(self.embedding, "execute"):
            encoded = self.embedding.execute(docs=docs)
        else:
            encoded = self.embedding.encode(docs)
        return np.asarray(encoded)
