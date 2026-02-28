from src.base.pipeline import Pipelines
from src.pipeline.rigid_trajectory import RigidTrajectoryPipeline
from src.pipeline.rigid_trajectory_image import RigidTrajectoryImagePipeline
from src.pipeline.tool_call import ToolCallPipeline
from src.pipeline.tool_call_abstraction import DSPyToolCallPipelineAbstraction
from src.pipeline.tool_call_curated import DSPyToolCallPipelineCurated
from src.pipeline.tool_call_full import DSPyToolCallPipelineFull
from src.pipeline.tool_call_image import ToolCallImagePipeline

pipelines = {
    Pipelines.RIGID_TRAJECTORY: RigidTrajectoryPipeline,
    Pipelines.RIGID_TRAJECTORY_IMAGE: RigidTrajectoryImagePipeline,
    Pipelines.TOOL_CALL: ToolCallPipeline,
    Pipelines.TOOL_CALL_IMAGE: ToolCallImagePipeline,
    Pipelines.TOOL_CALL_ABSTRACTION: DSPyToolCallPipelineAbstraction,
    Pipelines.TOOL_CALL_CURATED: DSPyToolCallPipelineCurated,
    Pipelines.TOOL_CALL_FULL: DSPyToolCallPipelineFull,
}
