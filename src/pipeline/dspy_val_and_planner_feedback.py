import logging
from pathlib import Path

import dspy

from src.base.pipeline import CombinedMeta, Pipelines
from src.base.schema import PDDLFiles, PipelineError
from src.constants import project_root
from src.eval.fast_downward import ExitCodes, FDErrorInfo
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference import Models
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, domain_prompts, get_prompt, problem_prompts

logger = logging.getLogger(__name__)


# TODO: Think about moving these signatures to the module.
#       this might enable better reuse but also might not
class GenerateDomain(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT)

    task_description: str = dspy.InputField(
        desc="Natural language description of the planning domain to model"
    )
    domain_pddl: str = dspy.OutputField(
        desc="Complete PDDL domain file adhering to PDDL 1.0 standard"
    )


class GenerateProblem(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT)

    domain_pddl: str = dspy.InputField(desc="The previously generated PDDL domain file")
    problem_description: str = dspy.InputField(
        desc="Natural language description of the specific problem instance"
    )
    problem_pddl: str = dspy.OutputField(
        desc="Complete PDDL problem file adhering to PDDL 1.0 standard"
    )


class FixDomainSyntax(dspy.Signature):
    __doc__ = get_prompt(Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN)

    domain_pddl: str = dspy.InputField(
        desc="The PDDL domain file containing syntax errors"
    )
    errors: str = dspy.InputField(
        desc="List of syntax errors with line numbers and error messages"
    )
    fixed_domain_pddl: str = dspy.OutputField(
        desc="The corrected PDDL domain file with all syntax errors fixed"
    )


class FixProblemSyntax(dspy.Signature):
    __doc__ = get_prompt(Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_PROBLEM)

    domain_pddl: str = dspy.InputField(
        desc="The syntactically correct PDDL domain file for context"
    )
    problem_pddl: str = dspy.InputField(
        desc="The PDDL problem file containing syntax errors"
    )
    errors: str = dspy.InputField(
        desc="List of syntax errors with line numbers and error messages"
    )
    fixed_problem_pddl: str = dspy.OutputField(
        desc="The corrected PDDL problem file with all syntax errors fixed"
    )


class FixUnsolvablePDDL(dspy.Signature):
    __doc__ = get_prompt(Prompts.PLANNER_CONTEXT, Prompts.PLANNER_TASK)

    file_to_fix: str = dspy.InputField(desc="Which file to fix: 'domain' or 'problem'")
    domain_pddl: str = dspy.InputField(desc="The current PDDL domain file")
    problem_pddl: str = dspy.InputField(desc="The current PDDL problem file")
    fixed_pddl: str = dspy.OutputField(
        desc="The fixed PDDL file (domain or problem as specified)"
    )


class FixTranslateError(dspy.Signature):
    __doc__ = get_prompt(
        Prompts.PLANNER_TRANSLATE_CONTEXT, Prompts.PLANNER_TRANSLATE_TASK
    )

    file_to_fix: str = dspy.InputField(desc="Which file to fix: 'domain' or 'problem'")
    error_message: str = dspy.InputField(
        desc="The error message from the planner's translator"
    )
    file_content: str = dspy.InputField(desc="The content of the file to fix")
    fixed_pddl: str = dspy.OutputField(
        desc="The fixed PDDL file with the translation error resolved"
    )


class DSPyValAndPlannerFeedbackPipeline(
    dspy.Module, ValFeedbackPipeline, metaclass=CombinedMeta
):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        dspy.Module.__init__(self)
        ValFeedbackPipeline.__init__(
            self, model, domain, pipeline or Pipelines.DSPY_VAL_AND_PLANNER_FEEDBACK
        )

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
        dspy.configure(lm=self.lm)

        self.generate_domain_module = dspy.ChainOfThought(GenerateDomain)
        self.generate_problem_module = dspy.ChainOfThought(GenerateProblem)
        self.fix_domain_module = dspy.ChainOfThought(FixDomainSyntax)
        self.fix_problem_module = dspy.ChainOfThought(FixProblemSyntax)
        self.fix_unsolvable_module = dspy.ChainOfThought(FixUnsolvablePDDL)
        self.fix_translate_module = dspy.ChainOfThought(FixTranslateError)

    def fix_domain(
        self,
        domain_file: Path,
        num_tries: int = 5,
        image_paths: list[str] | None = None,
    ):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_domain(domain_file)
            self.domain_syntax_errors_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations domain syntax fixes: {i}")
                domain_content = self._read_pddl_file(domain_file)
                errors = err_info.get_lines_with_errors()

                self.num_model_calls += 1
                result = self.fix_domain_module(
                    domain_pddl=domain_content, errors=errors
                )
                domain_file = self._write_pddl_file(
                    result.fixed_domain_pddl, file=domain_file
                )

    def fix_problem(
        self,
        domain_file: Path,
        problem_file: Path,
        num_tries: int = 5,
        image_paths: list[str] | None = None,
    ):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            self.problem_syntax_mistakes_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations problem syntax fixes: {i}")
                domain_content = self._read_pddl_file(domain_file)
                problem_content = self._read_pddl_file(problem_file)
                errors = err_info.get_lines_with_errors()

                self.num_model_calls += 1
                result = self.fix_problem_module(
                    domain_pddl=domain_content,
                    problem_pddl=problem_content,
                    errors=errors,
                )
                problem_file = self._write_pddl_file(
                    result.fixed_problem_pddl, file=problem_file
                )

    def fix_plan_not_found(self, domain_file: Path, problem_file: Path):
        domain_content = self._read_pddl_file(domain_file)
        problem_content = self._read_pddl_file(problem_file)

        self.num_model_calls += 1
        domain_result = self.fix_unsolvable_module(
            file_to_fix=PDDLFiles.DOMAIN,
            domain_pddl=domain_content,
            problem_pddl=problem_content,
        )
        self._write_pddl_file(domain_result.fixed_pddl, file=domain_file)

        domain_content = self._read_pddl_file(domain_file)
        self.num_model_calls += 1
        problem_result = self.fix_unsolvable_module(
            file_to_fix=PDDLFiles.PROBLEM,
            domain_pddl=domain_content,
            problem_pddl=problem_content,
        )
        self._write_pddl_file(problem_result.fixed_pddl, file=problem_file)

    def fix_parsing_error(
        self, domain_file: Path, problem_file: Path, planner_output: FDErrorInfo
    ):
        if planner_output.file == PDDLFiles.DOMAIN:
            content = self._read_pddl_file(domain_file)
        else:
            content = self._read_pddl_file(problem_file)

        self.num_model_calls += 1
        result = self.fix_translate_module(
            file_to_fix=str(planner_output.file),
            error_message=planner_output.error_message,
            file_content=content,
        )

        if planner_output.file == PDDLFiles.DOMAIN:
            self._write_pddl_file(result.fixed_pddl, file=domain_file)
        else:
            self._write_pddl_file(result.fixed_pddl, file=problem_file)

    def fix_planning(
        self,
        domain_file: Path,
        problem_file: Path,
        num_tries: int = 5,
    ) -> FDErrorInfo | Path:
        assert num_tries > 0
        iterations = 0
        planner_output = None
        for i in range(num_tries):
            planner_output = self._generate_plan(domain_file, problem_file)
            self.generate_plan_calls += 1
            if type(planner_output) is str:
                break
            elif type(planner_output) is FDErrorInfo and planner_output.exit_code in [
                ExitCodes.TRANSLATE_CRITICAL_ERROR,
                ExitCodes.TRANSLATE_INPUT_ERROR,
            ]:
                logger.debug("Parsing error")
                assert planner_output.file is not None
                iterations = i + 1
                self.fix_parsing_error(domain_file, problem_file, planner_output)
            else:
                logger.debug("Planning error")
                iterations = i + 1
                self.fix_plan_not_found(domain_file, problem_file)
            logger.debug(f"Iterations planning fixes: {iterations}")
        assert planner_output is not None
        return planner_output

    def forward(self) -> dspy.Prediction:
        task_prompt = domain_prompts[self.domain]

        self.num_model_calls += 1
        domain_result = self.generate_domain_module(task_description=task_prompt)
        domain = domain_result.domain_pddl
        self.print_and_clear_history()

        self.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        self.fix_domain(self.domain_file)
        if not self.is_domain_valid(self.domain_file):
            self.print_and_clear_history()
            logger.debug("Domain failure")
            return dspy.Prediction(out=PipelineError.DOMAIN_FAILURE)
        self.print_and_clear_history()

        problem_prompt = problem_prompts[self.domain]
        self.num_model_calls += 1
        problem_result = self.generate_problem_module(
            domain_pddl=domain, problem_description=problem_prompt
        )
        problem = problem_result.problem_pddl
        self.print_and_clear_history()

        self.problem_file = self._write_pddl_file(
            problem, pddl_file_type=PDDLFiles.PROBLEM
        )
        self.fix_problem(self.domain_file, self.problem_file)
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            self.print_and_clear_history()
            logger.debug("Problem failure")
            return dspy.Prediction(out=PipelineError.PROBLEM_FAILURE)
        self.print_and_clear_history()

        planner_output = self.fix_planning(self.domain_file, self.problem_file)
        if isinstance(planner_output, FDErrorInfo):
            self.print_and_clear_history()
            logger.debug("Failed to generate a plan")
            return dspy.Prediction(out=planner_output.to_pipeline_error())
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        self.print_and_clear_history()
        return dspy.Prediction(out=None)

    def print_and_clear_history(self):
        # Log interaction history in one shot and clear everything after
        # to always get interactions from last run only
        logger.debug(self.lm.inspect_history())
        self.lm.history.clear()

    def compile_module(self):
        # TODO: Implement one of the dspy optimization functions for this module here.
        #       The function can take old PDDL problems and descriptions as input.
        #       -> Either take them from old competitions or look into one of the existing
        #          LLM PDDL generation benchmarking papers.
        ...

    def _run_impl(self):
        prediction = self()
        return self.create_result(error=prediction.out)
