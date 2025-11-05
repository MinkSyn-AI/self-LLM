import os
from typing import Optional, Type

from openai import OpenAI

from ..base import ModuleEngine
from .generator_base import GeneratorType


class vLLMGeneratorEngine(ModuleEngine):
    generator_type: GeneratorType = GeneratorType.OFFLINE

    @classmethod
    def from_builded(
        cls,
        name: str = "model",
        api_key: Optional[str] = None,
        ip: str = "localhost",
        port: int = 8000,
        **kwargs,
    ) -> Type[ModuleEngine]:
        try:
            if api_key is None:
                api_key = os.getenv('VLLM_API_KEY', '0')

            cls.name = name
            cls.addr = f"http://{ip}:{port}/v1"
            cls.client = OpenAI(api_key=api_key, base_url=cls.addr)

        except Exception as e:
            raise ValueError(f"vLLM failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, prompt: str, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.name, messages=prompt
            )
            return response.choices[0].message.content

        except Exception as e:
            raise ValueError(
                f"Failed to calling API with vLLM. Error details: {e}"
            ) from e
