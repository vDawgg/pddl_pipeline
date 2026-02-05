import logging
import shutil
import time
import traceback
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum, StrEnum, auto
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from typing import TypeVar

import dspy
import polars as pl
from pydantic import BaseModel
from tqdm import tqdm

from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.constants import generated_pddl_dir, logs_dir, plans_dir, results_dir
from src.eval.fast_downward import ExitCodes, FDErrorInfo, exit_codes, parse_error
from src.inference import Models
from src.utils.domains import Domains
from src.utils.logger import add_file_handler, remove_file_handler
from src.utils.timestamp import get_current_timestamp

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


_ProgramMeta = type(dspy.Module)


class CombinedMeta(_ProgramMeta, ABCMeta):
    """Metaclass combining dspy's ProgramMeta with ABCMeta for multiple inheritance."""

    ...


class Pipelines(StrEnum):
    BASELINE = auto()
    VAL_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK = auto()
    VAL_AND_PLANNER_FEEDBACK_IMAGE = auto()
    DSPY_VAL_AND_PLANNER_FEEDBACK = auto()
    TOOL_CALL = auto()
    TOOL_CALL_IMAGE = auto()
    TOOL_CALL_MULTI_AGENT = auto()
    DSPY_TOOL_CALL = auto()


class PipelineBase(ABC):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        self.model = model
        # TODO: Just add these prompts according to the domain to this class. The dicts are too much.
        self.domain = domain
        self.pipeline = pipeline
        self.name = f"{self.domain}_{self.pipeline}_{self.model.split('/')[-1]}"
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
        return PipelineResult(
            model=self.model,
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
            logger.debug("Caught exception while running inference:")
            logger.debug(e)
            logger.debug(traceback.print_exc())
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

    def _save_plan(self, plan_file: Path) -> Path:
        latest_plan = ""
        for i in range(1, 100):
            candidate_path = Path(f"{plan_file}.{i}")
            if candidate_path.is_file():
                latest_plan = candidate_path
            else:
                break
        plan_name = plans_dir / f"{self.name}_{get_current_timestamp()}.plan"
        shutil.copyfile(latest_plan, plan_name)
        return plan_name

    def _generate_plan(
        self,
        domain_file: Path,
        problem_file: Path,
    ) -> FDErrorInfo | Path:
        plan_file = NamedTemporaryFile(delete=False)
        process = run(
            [
                "python",
                "../fast-downward-24.06.1/fast-downward.py",
                "--overall-time-limit",
                "1m",
                "--plan-file",
                plan_file.name,
                "--alias",
                "seq-sat-lama-2011",
                domain_file,
                problem_file,
            ],
            capture_output=True,
            text=True,
        )
        fd_code = exit_codes[process.returncode]
        if (
            fd_code == ExitCodes.SUCCESS
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME
        ):
            return self._save_plan(Path(plan_file.name))
        else:
            return parse_error(fd_code, domain_file, problem_file)

    def _write_pddl_file(
        self,
        pddl: str,
        file: Path | None = None,
        pddl_file_type: PDDLFiles | None = None,
    ) -> Path:
        file_path = (
            file
            or generated_pddl_dir
            / f"{self.domain}_{pddl_file_type}_{self.name}_{get_current_timestamp()}.pddl"
        )
        with open(file_path, "w") as f:
            f.write(pddl)
        return file_path

    def _read_pddl_file(self, pddl_file: Path) -> str:
        with open(pddl_file) as f:
            return f.read()
