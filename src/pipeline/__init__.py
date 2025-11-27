from enum import StrEnum

from src.pipeline.baseline import Baseline


class Pipelines(StrEnum):
    BASELINE = "baseline"


pipelines = {Pipelines.BASELINE: Baseline}
