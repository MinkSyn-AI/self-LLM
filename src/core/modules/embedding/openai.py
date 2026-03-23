import os
from typing import List, Optional, Type

import openai

from ..base import ModuleEngine
from .embed_base import EmbeddingType


class OpenAIEmbeddingEngine(ModuleEngine):
    embed_type: EmbeddingType = EmbeddingType.API

    @classmethod
    def from_builded(
        cls,
        name: str = "text-embedding-3-small",
        dimensions: int = 768,
        apiKey: Optional[str] = None,
        orgId: Optional[str] = None,
        **kwargs,
    ) -> Type[ModuleEngine]:
        cls.name = name
        cls.dimensions = dimensions
        cls.apiKey = apiKey or os.getenv("OPENAI_API_KEY")
        cls.orgId = orgId or os.getenv("OPENAI_ORG_ID")
        cls.baseUrl = kwargs.get("baseUrl") or os.getenv("OPENAI_BASE_URL")

        if not cls.apiKey:
            raise ValueError("The OpenAI API key must not be 'None'.")

        try:
            cls.client = openai.Client(
                base_url=cls.baseUrl, api_key=cls.apiKey, organization=cls.orgId
            )

        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e

        return cls()

    def execute(self, docs: List[str], **kwargs):
        try:
            embeds = self.client.embeddings.create(
                input=docs,
                model=self.name,
                dimensions=self.dimensions,
            )
            embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
            return embeddings

        except Exception as e:
            raise ValueError(f"Failed to get embeddings. Error details: {e}") from e
