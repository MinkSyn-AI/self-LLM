from typing import Dict

from ..base import BaseModule, ModuleEngine
from ..generate import AutoGeneratorModule
from .cross_encoder import CrossEncoderRerankEngine
from .rerank_base import RerankName


class AutoRerankModule(BaseModule):
    _module_mapping: Dict[RerankName, ModuleEngine] = {
        RerankName.CrossEncoder: CrossEncoderRerankEngine,
        RerankName.LLM: AutoGeneratorModule,
    }
