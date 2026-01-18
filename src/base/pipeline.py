import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
from pydantic import BaseModel
from tqdm import tqdm

from src.base.schema import PipelineError, PipelineResult
from src.constants import logs_dir, results_dir
from src.inference import Models
from src.inference.model_comm import (
    make_react_workflow as _make_react_workflow,
)
from src.inference.model_comm import (
    make_request as _make_request,
)
from src.utils.domains import Domains
from src.utils.logger import add_file_handler, remove_file_handler
from src.utils.timestamp import get_current_timestamp

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK = auto()
    TOOL_CALL = auto()
    TOOL_CALL_MULTI_AGENT = auto()


class PipelineBase(ABC):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        self.model = model
        # TODO: Just add these prompts according to the domain to this class. The dicts are too much.
        self.domain = domain
        self.pipeline = pipeline
        self.name = f"{self.domain}_{self.pipeline}_{self.model}"
        self.elapsed_time: float = 0.0
        self.num_model_calls: int = 0
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0
        self.domain_syntax_errors_calls = 0
        self.problem_syntax_mistakes_calls = 0
        self.generate_plan_calls = 0
        self.domain_file: Path | None = None
        self.problem_file: Path | None = None
        self.plan_file: Path | None = None
        self.log_file: Path | None = None

    def create_result(
        self,
        error: PipelineError | None = None,
    ) -> PipelineResult:
        res = PipelineResult(
            elapsed_time=self.elapsed_time,
            num_model_calls=self.num_model_calls,
            error=error,
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            plan_file=self.plan_file,
            log_file=self.log_file,
            create_pddl_file_calls=self.create_pddl_file_calls,
            read_pddl_file_calls=self.read_pddl_file_calls,
            edit_lines_calls=self.edit_lines_calls,
            domain_syntax_errors_calls=self.domain_syntax_errors_calls,
            problem_syntax_mistakes_calls=self.problem_syntax_mistakes_calls,
            generate_plan_calls=self.generate_plan_calls,
        )

        return res

    def run(self) -> PipelineResult:
        self.num_model_calls = 0
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0
        self.domain_syntax_errors_calls = 0
        self.problem_syntax_mistakes_calls = 0
        self.generate_plan_calls = 0
        self.domain_file = None
        self.problem_file = None
        self.plan_file = None
        self.log_file = logs_dir / f"{self.name}_{get_current_timestamp()}.log"
        file_handler = add_file_handler(self.log_file)
        start = time.perf_counter()
        try:
            result = self._run_impl()
        except Exception as e:
            self.elapsed_time = time.perf_counter() - start
            logger.debug(f"Caught exception while running inference: {e}")
            result = self.create_result(error=PipelineError.MODEL_FAILURE)
        else:
            self.elapsed_time = time.perf_counter() - start
        finally:
            remove_file_handler(file_handler)
        return result

    @abstractmethod
    def _run_impl(self) -> PipelineResult:
        pass

    def make_request(
        self,
        input_prompt: str,
        messages: list[Any] | None = None,
        format: type[T] | None = None,
        imgs: list[str] | None = None,
    ) -> tuple[T | str, list[Any]]:
        self.num_model_calls += 1
        return _make_request(
            input_prompt,
            model_name=self.model,
            messages=messages,
            format=format,
            imgs=imgs,
        )

    def make_react_workflow(
        self,
        input_prompt: str,
        tools: list[Callable],
        max_iters: int = 10,
    ) -> str:
        result, num_calls = _make_react_workflow(
            model_name=self.model,
            input_prompt=input_prompt,
            tools=tools,
            max_iters=max_iters,
        )
        self.num_model_calls += num_calls
        return result

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
