from enum import StrEnum, auto
from dataclasses import dataclass


class PDDLFiles(StrEnum):
    DOMAIN = auto()
    PROBLEM = auto()


class PipelineError(StrEnum):
    DOMAIN_FAILURE = auto()
    PROBLEM_FAILURE = auto()
    PLAN_FAILURE = auto()


@dataclass
class Action:
    action_name: str
    parameters: list[str]


@dataclass
class Plan:
    action_sequence: list[Action]
