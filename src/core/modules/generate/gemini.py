import os
from typing import Optional, Type

from google import genai

from ..base import ModuleEngine
from .generator_base import GeneratorType


class GeminiGeneratorEngine(ModuleEngine):
    generator_type: GeneratorType = GeneratorType.ONLINE

    @classmethod
    def from_builded(
        cls, name: str = "gemini-2.5-flash", api_key: Optional[str] = None, **kwargs
    ) -> Type[ModuleEngine]:
        try:
            if api_key is None:
                api_key = os.getenv('GOOGLE_API_KEY', '')

            cls.name = name
            cls.client = genai.Client(api_key=api_key)

        except Exception as e:
            raise ValueError(f"Gemini failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, prompt: str, **kwargs):
        try:
            response = self.client.models.generate_content(
                model=self.name,
                contents=[prompt],
            )

            return response.text

        except Exception as e:
            raise ValueError(
                f"Failed to calling API with Gemini. Error details: {e}"
            ) from e
