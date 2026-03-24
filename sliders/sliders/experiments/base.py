from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from sliders.system import System


class Experiment(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def _run_row(self, row: dict, system: System, all_metadata: list, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    async def run(self, system: System, parallel: bool = False, *args, **kwargs) -> dict:
        pass
