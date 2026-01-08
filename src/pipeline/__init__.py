from src.base.pipeline import Pipelines
from src.pipeline.baseline import Baseline
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline


pipelines = {
    Pipelines.BASELINE: Baseline,
    Pipelines.VAL_FEEDBACK: ValFeedbackPipeline,
    Pipelines.VAL_AND_PLANNER_FEEDBACK: ValAndPlannerFeedbackPipeline,
}
