from abc import ABC, abstractmethod

from src.inference import Models


class PipelineBase(ABC):
    def __init__(self, model: Models):
        self.model = model

    @abstractmethod
    def run(self) -> tuple[str, str]:
        pass
