import os
from typing import Optional, Type

import openai

from ..base import ModuleEngine
from .generator_base import GeneratorType


class OpenAIGeneratorEngine(ModuleEngine):
    generator_type: GeneratorType = GeneratorType.ONLINE

    @classmethod
    def from_builded(
        cls, name: str = "chat-gpt", api_key: Optional[str] = None, **kwargs
    ) -> Type[ModuleEngine]:
        try:
            if api_key is None:
                api_key = os.getenv('OPENAPI_API_KEY', '')

            cls.name = name
            cls.client = openai.OpenAI(api_key=api_key)

        except Exception as e:
            raise ValueError(f"OpenAI failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, prompt: str, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.name, messages=prompt
            )
            return response.choices[0].message.content

        except Exception as e:
            raise ValueError(
                f"Failed to calling API with OpenAI. Error details: {e}"
            ) from e
