import logging
from abc import ABC, abstractmethod
from enum import Enum, StrEnum, auto
from functools import wraps
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.base.schema import PipelineError, PipelineResult
from src.constants import results_dir
from src.inference import Models
from src.utils.domains import Domains
from src.utils.timestamp import get_current_timestamp

logger = logging.getLogger(__name__)


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK = auto()
    TOOL_CALL = auto()


def catch_model_failure(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Caught exception while running inference: {e}")
            return PipelineResult(PipelineError.MODEL_FAILURE)

    return wrapper


class PipelineBase(ABC):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        self.model = model
        # TODO: Just add these prompts according to the domain to this class. The dicts are too much.
        self.domain = domain
        self.pipeline = pipeline
        self.name = f"{self.domain}_{self.pipeline}_{self.model}"

    @abstractmethod
    @catch_model_failure
    def run(self) -> PipelineResult:
        pass

    def run_eval(self, iterations: int) -> Path:
        results: list[PipelineResult] = []
        for _ in tqdm(range(iterations), "Running Evaluation"):
            results.append(self.run())
        results_name = results_dir / f"{self.name}_{get_current_timestamp()}.csv"

        def serialize_value(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, Path):
                return str(v)
            return v

        pl.DataFrame(
            {k: serialize_value(v) for k, v in result.__dict__.items()}
            for result in results
        ).write_csv(results_name)
        return results_name
