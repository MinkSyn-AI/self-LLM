from abc import ABC, abstractmethod


class BaseBlocker(ABC):
    def execute(self, **kwargs):
        kwargs = self._preprocess(**kwargs)
        kwargs = self._execute_base(**kwargs)
        kwargs = self._postrocess(**kwargs)
        return kwargs

    def _preprocess(self, **kwargs):
        return kwargs

    def _postrocess(self, **kwargs):
        return kwargs

    @abstractmethod
    def _execute_base(self, **kwargs):
        pass
