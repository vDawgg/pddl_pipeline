from abc import ABC, abstractmethod

from src.inference import Models
from src.base.schema import PipelineError, Plan


class PipelineBase(ABC):
    def __init__(self, model: Models):
        self.model = model

    @abstractmethod
    def run(self) -> PipelineError | Plan:
        pass
