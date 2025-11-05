from enum import auto

from strenum import LowercaseStrEnum


class EmbeddingType(LowercaseStrEnum):
    API = auto()  # Buiding or calling with API protocol
    LOCAL = auto()  # Building at local server
    CONFIG = auto()  # Using with Hugging Face


class EmbeddingName(LowercaseStrEnum):
    FastEmbedding = auto()
    Google = auto()
    OpenAI = auto()
    Transformers = auto()
