from enum import auto

from strenum import LowercaseStrEnum


class RerankName(LowercaseStrEnum):
    CrossEncoder = auto()
    LLM = auto()
