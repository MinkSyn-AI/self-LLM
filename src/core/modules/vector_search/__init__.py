from typing import Dict

from ..base import BaseModule, ModuleEngine
from .chromadb import ChormaDBVectorEngine
from .mongodb import MongoDBVectorEngine
from .qdrant import QdrantVectorEngine
from .vector_base import VectorSearchName


class AutoVectorSearchModule(BaseModule):
    _module_mapping: Dict[VectorSearchName, ModuleEngine] = {
        VectorSearchName.ChromaDB: ChormaDBVectorEngine,
        VectorSearchName.Qdrant: QdrantVectorEngine,
        VectorSearchName.MongoDB: MongoDBVectorEngine,
    }
