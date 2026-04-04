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
    FEEDBACK_INCORPORATION_FAILURE = auto()
    MODEL_FAILURE = auto()


@dataclass
class PipelineResult:
    model: str
    elapsed_time: float = 0.0
    num_model_calls: int = 0
    error: PipelineError | None = None
    domain_file: Path | None = None
    problem_file: Path | None = None
    plan_file: Path | None = None
    log_file: Path | None = None
    ablate_tools: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    create_pddl_file_calls: int = 0
    read_pddl_file_calls: int = 0
    edit_lines_calls: int = 0
    domain_syntax_errors_calls: int = 0
    problem_syntax_mistakes_calls: int = 0
    translate_pddl_calls: int = 0
    generate_plan_calls: int = 0
    get_plan_feedback_calls: int = 0


class Pipelines(StrEnum):
    TOOL_CALL = auto()
    TOOL_CALL_ABSTRACTION = auto()
    TOOL_CALL_CURATED = auto()
    TOOL_CALL_FULL = auto()
    TOOL_CALL_IMAGE = auto()
    RIGID_TRAJECTORY = auto()
    RIGID_TRAJECTORY_IMAGE = auto()


class Tools(StrEnum):
    CREATE_PDDL_FILE = auto()
    READ_PDDL_FILE = auto()
    EDIT_LINES = auto()
    GET_SYNTAX_MISTAKES_DOMAIN = auto()
    GET_SYNTAX_MISTAKES_PROBLEM = auto()
    TRANSLATE_PDDL = auto()
    GENERATE_PLAN = auto()
    GET_PLAN_FEEDBACK = auto()


class Prompts(StrEnum):
    DOMAIN = "domain.md"
    PROBLEM = "problem.md"
    DOMAIN_AND_PROBLEM = "domain_and_problem.md"
    GENERATION_CONTEXT = "generation_context.md"
    GENERATION_CONTEXT_TOOLS = "generation_context_tools.md"
    GENERATION_CONTEXT_TOOLS_IMAGES = "generation_context_tools_images.md"
    GENERATION_CONTEXT_IMAGES = "generation_context_images.md"
    GENERATION_CONTEXT_PLAN_FEEDBACK = "generation_context_plan_feedback.md"
    GENERATION_CONTEXT_ACTION_MAPPING = "generation_context_action_mapping.md"
    RING_AND_PEG_DOMAIN = "ring_and_peg.md"
    RING_AND_PEG_1 = "ring_and_peg_1.md"
    RING_AND_PEG_2 = "ring_and_peg_2.md"
    RING_AND_PEG_3 = "ring_and_peg_3.md"
    RING_AND_PEG_4 = "ring_and_peg_4.md"
    RING_AND_PEG_5 = "ring_and_peg_5.md"
    RING_AND_PEG_PLAN = "ring_and_peg_plan.md"
    NEEDLE_TRANSFER_DOMAIN = "needle_transfer_domain.md"
    NEEDLE_TRANSFER_1 = "needle_transfer_1.md"
    NEEDLE_TRANSFER_2 = "needle_transfer_2.md"
    VAL_FEEDBACK_CONTEXT = "val_feedback_context.md"
    VAL_FEEDBACK_CONTEXT_IMAGES = "val_feedback_context_images.md"
    VAL_FEEDBACK_DOMAIN = "val_feedback_domain.md"
    VAL_FEEDBACK_PROBLEM = "val_feedback_problem.md"
    PLANNER_CONTEXT = "planner_context.md"
    PLANNER_CONTEXT_IMAGES = "planner_context_images.md"
    PLANNER_TASK = "planner_task.md"
    PLANNER_TRANSLATE_CONTEXT = "planner_translate_context.md"
    PLANNER_TRANSLATE_CONTEXT_IMAGES = "planner_translate_context_images.md"
    PLANNER_TRANSLATE_TASK = "planner_translate_task.md"
    PLAN_FEEDBACK = "plan_feedback.md"
    ACTION_MAPPING = "action_mapping.md"
    ACTION_SCHEMA_RING_AND_PEG = "action_schema_ring_and_peg.md"
    ACTION_SCHEMA_NEEDLE_TRANSFER = "action_schema_needle_transfer.md"


class Domains(StrEnum):
    RING_AND_PEG = auto()
    NEEDLE_TRANSFER = auto()


class Problems(StrEnum):
    RING_AND_PEG_1 = auto()
    RING_AND_PEG_2 = auto()
    RING_AND_PEG_3 = auto()
    RING_AND_PEG_4 = auto()
    RING_AND_PEG_5 = auto()
    NEEDLE_TRANSFER_1 = auto()
    NEEDLE_TRANSFER_2 = auto()


class Images(StrEnum):
    RING_AND_PEG_1 = "ring_and_peg_1.png"
    RING_AND_PEG_2 = "ring_and_peg_2.png"
    RING_AND_PEG_3 = "ring_and_peg_3.png"
    RING_AND_PEG_4 = "ring_and_peg_4.png"
    RING_AND_PEG_5 = "ring_and_peg_5.png"
    NEEDLE_TRANSFER_1 = "needle_transfer_1.png"
    NEEDLE_TRANSFER_2 = "needle_transfer_2.png"
