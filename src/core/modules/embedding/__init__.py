from typing import Dict

from ..base import BaseModule, ModuleEngine
from .embed_base import EmbeddingName
from .fast_embed import FastEmbeddingEngine
from .google import GoogleEmbeddingEngine
from .openai import OpenAIEmbeddingEngine
from .transformer import TransformerEmbedding


class AutoEmbdeddingModule(BaseModule):
    _module_mapping: Dict[EmbeddingName, ModuleEngine] = {
        EmbeddingName.FastEmbedding: FastEmbeddingEngine,
        EmbeddingName.Google: GoogleEmbeddingEngine,
        EmbeddingName.OpenAI: OpenAIEmbeddingEngine,
        EmbeddingName.Transformers: TransformerEmbedding,
    }
