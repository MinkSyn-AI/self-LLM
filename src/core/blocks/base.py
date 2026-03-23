from abc import ABC, abstractmethod


class BaseBlocker(ABC):
    def execute(self, **kwargs):
        kwargs = self._preprocess(**kwargs)
        kwargs = self._execute_base(**kwargs)
        kwargs = self._postprocess(**kwargs)
        return kwargs

    def _preprocess(self, **kwargs):
        return kwargs

    def _postprocess(self, **kwargs):
        return kwargs

    @abstractmethod
    def _execute_base(self, **kwargs):
        pass
