from abc import ABC, abstractmethod
from enum import StrEnum, auto

from src.inference import Models
from src.base.schema import PipelineError
from src.utils.domains import Domains


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK = auto()


class PipelineBase(ABC):
    def __init__(self, model: Models, domain: Domains):
        self.model = model
        # TODO: Just add these prompts according to the domain to this class. The dicts are too much.
        self.domain = domain

    @abstractmethod
    def run(self) -> PipelineError | None:
        pass
