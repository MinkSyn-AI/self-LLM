from enum import auto

from strenum import LowercaseStrEnum


class VectorSearchName(LowercaseStrEnum):
    Qdrant = auto()
    MongoDB = auto()
    ChromaDB = auto()
