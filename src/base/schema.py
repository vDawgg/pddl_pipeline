from dataclasses import dataclass
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
    error: PipelineError | None = None
    domain_file: Path | None = None
    problem_file: Path | None = None
    plan_file: Path | None = None
    num_domain_fixes: int | None = None
    num_problem_fixes: int | None = None
    num_planner_fixes: int | None = None
