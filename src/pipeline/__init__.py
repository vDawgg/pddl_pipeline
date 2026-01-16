from src.base.pipeline import Pipelines
from src.pipeline.baseline import Baseline
from src.pipeline.tool_calls import ToolCallPipeline
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.pipeline.val_feedback import ValFeedbackPipeline

pipelines = {
    Pipelines.BASELINE: Baseline,
    Pipelines.VAL_FEEDBACK: ValFeedbackPipeline,
    Pipelines.VAL_AND_PLANNER_FEEDBACK: ValAndPlannerFeedbackPipeline,
    Pipelines.TOOL_CALL: ToolCallPipeline,
}
