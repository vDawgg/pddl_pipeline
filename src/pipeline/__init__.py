from enum import StrEnum, auto

from src.pipeline.baseline import Baseline
from src.pipeline.val_feedback import ValFeedbackPipeline


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()


pipelines = {
    Pipelines.BASELINE: Baseline,
    Pipelines.VAL_FEEDBACK: ValFeedbackPipeline,
}
