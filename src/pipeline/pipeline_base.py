from abc import ABC, abstractmethod
from enum import StrEnum, auto

from src.inference import Models
from src.eval.fast_downward import Plan


class PipelineError(StrEnum):
    DOMAIN_FAILURE = auto()
    PROBLEM_FAILURE = auto()
    PLAN_FAILURE = auto()


class PipelineBase(ABC):
    def __init__(self, model: Models):
        self.model = model

    @abstractmethod
    def run(self) -> PipelineError | Plan:
        pass
