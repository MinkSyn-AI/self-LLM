from enum import auto

from strenum import LowercaseStrEnum


class GeneratorType(LowercaseStrEnum):
    ONLINE = auto()  # Calling with API protocol
    OFFLINE = auto()  # Building at local server


class GeneratorName(LowercaseStrEnum):
    Gemini = auto()
    OpenAI = auto()

    vLLM = auto()
    Transformers = auto()
