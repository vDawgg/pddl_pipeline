from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path


class PDDLFiles(StrEnum):
    DOMAIN = auto()
    PROBLEM = auto()


class PipelineError(StrEnum):
    DOMAIN_FAILURE = auto()
    PROBLEM_FAILURE = auto()
    PLAN_FAILURE = auto()
    MODEL_FAILURE = auto()


@dataclass
class PipelineResult:
    elapsed_time: float
    error: PipelineError | None = None
    domain_file: Path | None = None
    problem_file: Path | None = None
    plan_file: Path | None = None
    num_domain_fixes: int | None = None
    num_problem_fixes: int | None = None
    num_planner_fixes: int | None = None
    _number_of_fixes: int | None = field(default=None, repr=False)

    @property
    def number_of_fixes(self) -> int | None:
        if self._number_of_fixes is not None:
            return self._number_of_fixes
        counts = [self.num_domain_fixes, self.num_problem_fixes, self.num_planner_fixes]
        if all(c is None for c in counts):
            return None
        return sum(c or 0 for c in counts)

    @number_of_fixes.setter
    def number_of_fixes(self, value: int | None) -> None:
        self._number_of_fixes = value
