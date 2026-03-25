from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import ConstructionResult


class BaseConstructor(ABC):
    nombre = "base_constructor"

    @abstractmethod
    def build(self, universe, name: str, **kwargs) -> ConstructionResult:
        """
        Construye una cartera sobre un universe compartido y devuelve el resultado.
        """
