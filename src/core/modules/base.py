from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from strenum import LowercaseStrEnum


class ModuleEngine(ABC):

    @classmethod
    @abstractmethod
    def from_builded(cls, *args, **kwargs):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def is_healthy(self) -> bool:
        return True


class BaseModule(ModuleEngine):
    engine_name: Optional[LowercaseStrEnum] = None

    _module: Optional[ModuleEngine] = None
    _module_mapping: Dict[LowercaseStrEnum, Type[ModuleEngine]] = {}

    @classmethod
    def from_builded(
        cls, engine_name: LowercaseStrEnum, *args, **kwargs
    ) -> ModuleEngine:
        if engine_name not in cls._module_mapping:
            raise ValueError(f"Engine {engine_name} not found in ModuleNames")

        instance = cls()
        instance.engine_name = engine_name
        instance._module = cls._module_mapping.get(engine_name)
        instance._module = instance._module.from_builded(*args, **kwargs)

        return instance

    def execute(self, *args, **kwargs):
        return self._module.execute(*args, **kwargs)

    def is_healthy(self) -> bool:
        if self._module is None:
            return False
        return self._module.is_healthy()
