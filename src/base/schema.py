from dataclasses import dataclass, field
from enum import StrEnum, auto


class PDDLFiles(StrEnum):
    DOMAIN = auto()
    PROBLEM = auto()


class PipelineError(StrEnum):
    DOMAIN_FAILURE = auto()
    PROBLEM_FAILURE = auto()
    PLAN_FAILURE = auto()


@dataclass
class PipelineResult:
    error: PipelineError | None = None
    iterations: dict[str, int] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None
