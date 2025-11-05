from typing import Dict

from ..base import BaseModule, ModuleEngine
from .gemini import GeminiGeneratorEngine
from .generator_base import GeneratorName
from .openai import OpenAIGeneratorEngine
from .transformer import TransformerGeneratorEngine
from .vllm import vLLMGeneratorEngine


class AutoGeneratorModule(BaseModule):
    _module_mapping: Dict[GeneratorName, ModuleEngine] = {
        GeneratorName.Gemini: GeminiGeneratorEngine,
        GeneratorName.vLLM: vLLMGeneratorEngine,
        GeneratorName.OpenAI: OpenAIGeneratorEngine,
        GeneratorName.Transformers: TransformerGeneratorEngine,
    }
