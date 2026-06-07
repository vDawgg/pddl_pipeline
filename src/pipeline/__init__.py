from src.base.pipeline import PipelineBase, Pipelines
from src.pipeline.rigid_trajectory import RigidTrajectoryPipeline
from src.pipeline.tool_call import ToolCallPipeline
from src.pipeline.tool_call_abstraction import ToolCallPipelineAbstraction
from src.pipeline.tool_call_curated import ToolCallPipelineCurated
from src.pipeline.tool_call_full import ToolCallPipelineFull
from src.pipeline.tool_call_image import ToolCallImagePipeline

pipelines: dict[Pipelines, type[PipelineBase]] = {
    Pipelines.RIGID_TRAJECTORY: RigidTrajectoryPipeline,
    Pipelines.TOOL_CALL: ToolCallPipeline,
    Pipelines.TOOL_CALL_IMAGE: ToolCallImagePipeline,
    Pipelines.TOOL_CALL_ABSTRACTION: ToolCallPipelineAbstraction,
    Pipelines.TOOL_CALL_CURATED: ToolCallPipelineCurated,
    Pipelines.TOOL_CALL_FULL: ToolCallPipelineFull,
}
