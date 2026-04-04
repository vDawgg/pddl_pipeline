import copy
import logging
import shutil
import threading
import time
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import dspy
import polars as pl
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from pydantic import BaseModel
from tqdm import tqdm

from src.base.schemas import (
    Domains,
    PDDLFiles,
    PipelineError,
    PipelineResult,
    Pipelines,
    Problems,
    Tools,
)
from src.constants import (
    generated_pddl_dir,
    logs_dir,
    plans_dir,
    project_root,
    results_dir,
)
from src.eval.fast_downward import (
    ExitCodes,
    FDErrorInfo,
    UnsolvabilityFeedback,
    generate_plan,
    parse_error,
)
from src.inference import Models, get_model_config
from src.utils.logger import add_file_handler, remove_file_handler
from src.utils.pddlgym_utils import goal_reached, make_ds
from src.utils.timestamp import get_current_timestamp

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ThreadSafeClassVars(threading.local):
    """
    Thread safe wrapper for class vars to keep track of the pipeline runs
    metrics and results. Needed, as dspy modules parallelization approach
    assumes pure functions and we cannot depend on that.
    """

    def __deepcopy__(self, memo):
        new = ThreadSafeClassVars()
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    def __copy__(self):
        new = ThreadSafeClassVars()
        for key, value in self.__dict__.items():
            setattr(new, key, value)
        return new

    def __init__(self):
        self.elapsed_time: float = 0.0
        self.num_model_calls: int = 0
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0
        self.domain_syntax_errors_calls = 0
        self.problem_syntax_mistakes_calls = 0
        self.translate_pddl_calls = 0
        self.generate_plan_calls = 0
        self.get_plan_feedback_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.domain_file: Path | None = None
        self.problem_file: Path | None = None
        self.plan_file: Path | None = None
        self.log_file: Path | None = None
        self.task_description: str | None = None

    def reset(self, name: str):
        self.num_model_calls = 0
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0
        self.domain_syntax_errors_calls = 0
        self.problem_syntax_mistakes_calls = 0
        self.translate_pddl_calls = 0
        self.generate_plan_calls = 0
        self.get_plan_feedback_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.domain_file = None
        self.problem_file = None
        self.plan_file = None
        self.log_file = logs_dir / f"{name}_{get_current_timestamp()}_{uuid4().hex}.log"
        self.task_description = None


class PipelineBase(dspy.Module):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        problem: Problems,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        self.model = model
        self.domain = domain
        self.problem = problem
        self.pipeline = pipeline
        self.name = f"{self.domain}_{self.problem}_{self.pipeline}_{self.model.split('/')[-1]}_base"
        self.vars = ThreadSafeClassVars()
        self.ablate_tools = ablate_tools

        self._model_config = get_model_config(self.model)
        key_path = project_root / self._model_config.key_file
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")
        api_key = key_path.read_text().strip().split("\n")[0]
        self.lm = dspy.LM(
            model="openai/" + self._model_config.api_model_name,
            api_key=api_key,
            api_base=self._model_config.base_url,
            cache=False,  # Disable DSPy's built-in response caching
        )
        dspy.configure(lm=self.lm, track_usage=True)
        if optimized_program is not None:
            assert Path(optimized_program).exists(), (
                "Path to optimized program does not exist"
            )
            self.name = f"{self.domain}_{self.problem}_{self.pipeline}_{self.model.split('/')[-1]}_{Path(optimized_program).name}"

    def deepcopy(self):
        new_instance = super().deepcopy()
        new_instance.vars = ThreadSafeClassVars()
        return new_instance

    ## PIPELINE RUN LOGIC

    def create_result(
        self,
        error: PipelineError | None = None,
    ) -> PipelineResult:
        return PipelineResult(
            model=self.model,
            elapsed_time=self.vars.elapsed_time,
            num_model_calls=self.vars.num_model_calls,
            error=error,
            domain_file=self.vars.domain_file,
            problem_file=self.vars.problem_file,
            plan_file=self.vars.plan_file,
            log_file=self.vars.log_file,
            ablate_tools=";".join(self.ablate_tools) if self.ablate_tools else None,
            input_tokens=self.vars.input_tokens,
            output_tokens=self.vars.output_tokens,
            create_pddl_file_calls=self.vars.create_pddl_file_calls,
            read_pddl_file_calls=self.vars.read_pddl_file_calls,
            edit_lines_calls=self.vars.edit_lines_calls,
            domain_syntax_errors_calls=self.vars.domain_syntax_errors_calls,
            problem_syntax_mistakes_calls=self.vars.problem_syntax_mistakes_calls,
            translate_pddl_calls=self.vars.translate_pddl_calls,
            generate_plan_calls=self.vars.generate_plan_calls,
            get_plan_feedback_calls=self.vars.get_plan_feedback_calls,
        )

    def run(self) -> PipelineResult:
        self.vars.reset(self.name)
        assert self.vars.log_file is not None
        file_handler = add_file_handler(self.vars.log_file)
        start = time.perf_counter()
        try:
            result = self._run_impl()
        except Exception as e:
            self.elapsed_time = time.perf_counter() - start
            logger.debug("Caught exception while running inference:")
            logger.debug(e)
            result = self.create_result(error=PipelineError.MODEL_FAILURE)
        else:
            self.vars.elapsed_time = time.perf_counter() - start
        finally:
            remove_file_handler(file_handler)
        result.elapsed_time = self.vars.elapsed_time
        return result

    @abstractmethod
    def _run_impl(self) -> PipelineResult:
        pass

    def run_eval(self, iterations: int) -> Path:
        results: list[PipelineResult] = []
        for _ in tqdm(range(iterations), desc="Running Evaluation"):
            results.append(self.run())
        results_name = (
            results_dir / f"{self.name}_{get_current_timestamp()}_{uuid4().hex}.csv"
        )

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

    ## PIPELINE OPTIMIZATION FUNCTIONS

    def _pddl_generation_metric(
        self,
        example: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ) -> ScoreWithFeedback:
        """Evaluation metric for usage with DSPy optimizers. Tries to optimize for incremental success."""
        pred_error: PipelineError | None = pred.out
        if pred_error == PipelineError.PLAN_FAILURE_TRANSLATE:
            return ScoreWithFeedback(
                score=0.25,
                feedback="The program was unable to generate PDDL that could be used by the planning system due to syntactic or structural issues.",
            )
        elif pred_error == PipelineError.PLAN_FAILURE_UNSOLVABLE:
            return ScoreWithFeedback(
                score=0.5,
                feedback="The program was unable to generate solvable PDDL",
            )
        elif pred_error is None:
            # We cannot use the class instances plan files here, as this metric function
            # is not multi-threaded and therefor only has the vars instance of the original
            # class instance.
            assert pred.plan_file is not None
            planning_success = goal_reached(
                example.domain_name,
                example.problem_index,
                pred.plan_file,
            )
            if planning_success:
                return ScoreWithFeedback(
                    score=1.0,
                    feedback="The program successfully generated PDDL that could be used to generate a plan solving the given task.",
                )
            else:
                return ScoreWithFeedback(
                    score=0.75,
                    feedback="The program generated solvable PDDL that resulted in a plan, which did not solve the given task.",
                )
        # TODO:Figure out how to differentiate this from the translation errors
        return ScoreWithFeedback(
            score=0.0,
            feedback="The program was unable to generate a syntactically valid domain/problem.",
        )

    def _optimize_program(self, separate_prompts: bool = False):
        log_file = logs_dir / f"{self.name}_optimization_{get_current_timestamp()}.log"
        file_handler = add_file_handler(log_file)
        start = time.perf_counter()
        logger.info(f"Starting optimization for {self.name}")

        reflection_config = get_model_config(Models.GPT_52)
        key_path = project_root / reflection_config.key_file
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")
        api_key = key_path.read_text().strip().split("\n")[0]

        teleprompter = dspy.GEPA(
            max_full_evals=4,
            metric=self._pddl_generation_metric,
            num_threads=10,
            reflection_minibatch_size=10,
            reflection_lm=dspy.LM(
                model="openai/gpt-5.2",
                api_key=api_key,
                api_base=reflection_config.base_url,
                temperature=1.0,
                cache=False,
            ),
        )

        trainset, valset = make_ds(separate_prompts=separate_prompts)
        optimized_program = teleprompter.compile(
            self,
            trainset=trainset,
            valset=valset,
        )

        program_name = f"optimized_{self.pipeline}_{self.model}.json"
        optimized_program.save(program_name)
        elapsed_time = time.perf_counter() - start
        logger.info(
            f"Saved optimized program under {program_name} (took {elapsed_time:.2f}s)"
        )

        # Shut down litellm's background logging executor gracefully to avoid
        # "cannot schedule new futures after shutdown" after optimization is done
        try:
            from litellm.litellm_core_utils.thread_pool_executor import (
                executor as litellm_executor,
            )

            litellm_executor.shutdown(wait=True)
        except Exception:
            pass

        remove_file_handler(file_handler)

    @abstractmethod
    def compile_module(self):
        pass

    ## FASTDOWNWARD CLASS UTILITIES

    def _save_plan(self, plan_file: Path) -> Path:
        latest_plan = None
        for i in range(1, 100):
            candidate_path = Path(f"{plan_file}.{i}")
            if candidate_path.is_file():
                latest_plan = candidate_path
            else:
                break
        if latest_plan is None:
            raise FileNotFoundError(
                f"Fast Downward reported success but no plan file found at {plan_file}.* "
            )
        plan_name = (
            plans_dir / f"{self.name}_{get_current_timestamp()}_{uuid4().hex}.plan"
        )
        shutil.copyfile(latest_plan, plan_name)
        return plan_name

    def _generate_plan(
        self,
        domain_file: Path,
        problem_file: Path,
        unsolvability_feedback: UnsolvabilityFeedback = UnsolvabilityFeedback.SIMPLE,
    ) -> FDErrorInfo | Path:
        plan_file, fd_code, output = generate_plan(domain_file, problem_file)
        if (
            fd_code == ExitCodes.SUCCESS
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME
            or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME
        ):
            return self._save_plan(plan_file)
        else:
            return parse_error(
                output, fd_code, domain_file, problem_file, unsolvability_feedback
            )

    ## PDDL I/O

    def _write_pddl_file(
        self,
        pddl: str,
        file: Path | None = None,
        pddl_file_type: PDDLFiles | None = None,
    ) -> Path:
        assert file or pddl_file_type, (
            "Specify either the file to write to or the type of the PDDL file"
        )
        file_path = (
            file
            or generated_pddl_dir
            / f"{self.domain}_{pddl_file_type}_{self.name}_{get_current_timestamp()}_{uuid4().hex}.pddl"
        )
        with open(file_path, "w") as f:
            f.write(pddl)
        return file_path

    def _read_pddl_file(self, pddl_file: Path) -> str:
        with open(pddl_file) as f:
            return f.read()

    ## GENERAL UTIL

    def log_and_clear_history(self):  # noqa: C901
        """
        Log interaction history in one shot and clear everything after to always get interactions from last run only
        """
        # Code below adapted from pretty_print_history from dspy
        item = self.history[-1]
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        for msg in messages:
            logger.debug(f"{msg['role'].capitalize()} message:")
            if isinstance(msg["content"], str):
                logger.debug(msg["content"].strip())
            else:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c["type"] == "text":
                            logger.debug(c["text"].strip())
                        elif c["type"] == "image_url":
                            image_str = ""
                            if "base64" in c["image_url"].get("url", ""):
                                len_base64 = len(
                                    c["image_url"]["url"].split("base64,")[1]
                                )
                                image_str = (
                                    f"<{c['image_url']['url'].split('base64,')[0]}base64,"
                                    f"<IMAGE BASE 64 ENCODED({len_base64!s})>"
                                )
                            else:
                                image_str = f"<image_url: {c['image_url']['url']}>"
                            logger.debug(image_str.strip())
        if isinstance(outputs[0], dict):
            if outputs[0]["text"]:
                logger.debug("Response:")
                logger.debug(outputs[0]["text"].strip())

            if outputs[0].get("tool_calls"):
                logger.debug("Tool calls:")
                for tool_call in outputs[0]["tool_calls"]:
                    logger.debug(
                        f"{tool_call['function']['name']}: {tool_call['function']['arguments']}"
                    )
        else:
            logger.debug("Response:")
            logger.debug(outputs[0].strip())

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            logger.debug(choices_text)
        self.lm.history.clear()
