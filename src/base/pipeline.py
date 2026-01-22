import logging
import time
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.base.schema import PipelineError, PipelineResult
from src.constants import logs_dir
from src.inference import Models
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
        self.translate_pddl_calls = 0
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
            translate_pddl_calls=self.translate_pddl_calls,
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
        result.elapsed_time = self.elapsed_time
        return result

    @abstractmethod
    def _run_impl(self) -> PipelineResult:
        pass
