import os
from typing import List, Optional, Type

from ..base import ModuleEngine
from .embed_base import EmbeddingType


class GoogleEmbeddingEngine(ModuleEngine):
    embed_type: EmbeddingType = EmbeddingType.API

    @classmethod
    def from_builded(
        cls,
        name: str = "textembedding-gecko@003",
        baseUrl: Optional[str] = None,
        apiKey: Optional[str] = None,
        projectId: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs,
    ) -> Type[ModuleEngine]:
        cls.name = name
        cls.baseUrl = baseUrl
        cls.apiKey = apiKey

        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel

        except ImportError:
            raise ImportError(
                "To use GoogleEmbedding, please install the Google Cloud and Vertex AI libraries "
                "You can do this with the following command: "
                "`pip install google-cloud-aiplatform google-generativeai`"
            )

        projectId = projectId or os.getenv("GOOGLE_PROJECT_ID")
        location = location or os.getenv("GOOGLE_LOCATION", "us-central1")
        baseUrl = baseUrl or os.getenv("GOOGLE_BASE_URL")

        if projectId is None:
            raise ValueError("Google Project ID cannot be null.")

        try:
            aiplatform.init(project=projectId, location=location, api_endpoint=baseUrl)
            cls.client = TextEmbeddingModel.from_pretrained(cls.name)

        except Exception as err:
            raise ValueError(
                f"Failed to initialize Google AI Platform client. Error: {err}"
            ) from err

        return cls

    def execute(self, docs: List[str], **kwargs):
        try:
            embeddings = self.client.get_embeddings(docs)
            return [embedding.values for embedding in embeddings]

        except Exception as e:
            raise ValueError(f"Google AI Platform API call failed. Error: {e}") from e
