import os
from typing import Optional, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import ModuleEngine
from .generator_base import GeneratorType


class TransformerGeneratorEngine(ModuleEngine):
    generator_type: GeneratorType = GeneratorType.OFFLINE

    @classmethod
    def from_builded(
        cls, name: str = "gemini-2.5-flash", api_key: Optional[str] = None, **kwargs
    ) -> Type[ModuleEngine]:
        try:
            cls.name = name
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"

            cls.tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
            )

            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token

            cls.client = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(cls.device)

            cls.client.eval()

        except Exception as e:
            raise ValueError(f"Gemini failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, prompt: str, **kwargs):
        try:
            messages = prompt
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # thinking mode unabled
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.client.device
            )

            with torch.no_grad():
                generated_ids = self.client.generate(
                    **model_inputs,
                    max_new_tokens=kwargs.get('max_token', 4096),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            response_data = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return response_data

        except Exception as e:
            raise ValueError(
                f"Failed to calling API with Gemini. Error details: {e}"
            ) from e
