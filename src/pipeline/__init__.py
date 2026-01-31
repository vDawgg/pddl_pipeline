from src.base.pipeline import Pipelines
from src.pipeline.baseline import Baseline
from src.pipeline.tool_calls import ToolCallPipeline
from src.pipeline.tool_calls_image import ToolCallImagePipeline
from src.pipeline.tool_calls_multi_agent import ToolCallPipelineMutltiAgent
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.pipeline.val_and_planner_feedback_image import (
    ValAndPLannerFeedbackImagePipeline,
)
from src.pipeline.val_feedback import ValFeedbackPipeline

pipelines = {
    Pipelines.BASELINE: Baseline,
    Pipelines.VAL_FEEDBACK: ValFeedbackPipeline,
    Pipelines.VAL_AND_PLANNER_FEEDBACK: ValAndPlannerFeedbackPipeline,
    Pipelines.VAL_AND_PLANNER_FEEDBACK_IMAGE: ValAndPLannerFeedbackImagePipeline,
    Pipelines.TOOL_CALL: ToolCallPipeline,
    Pipelines.TOOL_CALL_IMAGE: ToolCallImagePipeline,
    Pipelines.TOOL_CALL_MULTI_AGENT: ToolCallPipelineMutltiAgent,
}
