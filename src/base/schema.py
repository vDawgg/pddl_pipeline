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
    PLAN_FAILURE_TRANSLATE = auto()
    PLAN_FAILURE_UNSOLVABLE = auto()
    MODEL_FAILURE = auto()


@dataclass
class PipelineResult:
    elapsed_time: float = 0.0
    num_model_calls: int = 0
    error: PipelineError | None = None
    domain_file: Path | None = None
    problem_file: Path | None = None
    plan_file: Path | None = None
    log_file: Path | None = None
    create_pddl_file_calls: int = 0
    read_pddl_file_calls: int = 0
    edit_lines_calls: int = 0
    domain_syntax_errors_calls: int = 0
    problem_syntax_mistakes_calls: int = 0
    translate_pddl_calls: int = 0
    generate_plan_calls: int = 0
