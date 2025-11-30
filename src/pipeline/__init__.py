from enum import StrEnum, auto

from src.pipeline.baseline import Baseline
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK = auto()


pipelines = {
    Pipelines.BASELINE: Baseline,
    Pipelines.VAL_FEEDBACK: ValFeedbackPipeline,
    Pipelines.VAL_AND_PLANNER_FEEDBACK: ValAndPlannerFeedbackPipeline,
}
