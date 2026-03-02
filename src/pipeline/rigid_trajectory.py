import logging
from pathlib import Path

import dspy

from src.base.pipeline import PipelineBase, Pipelines
from src.base.schema import PDDLFiles, PipelineError
from src.eval.fast_downward import ExitCodes, FDErrorInfo
from src.eval.val import (
    get_syntax_mistakes_domain,
    get_syntax_mistakes_problem,
    is_domain_valid,
    is_problem_valid,
)
from src.inference import Models
from src.utils.domains import Domains
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class GenerateDomain(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT)

    domain_description: str = dspy.InputField(
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


class RigidTrajectoryPipeline(PipelineBase):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        super().__init__(
            model, domain, pipeline or Pipelines.RIGID_TRAJECTORY, optimized_program
        )
        self.generate_domain_module = dspy.Predict(GenerateDomain)
        self.generate_problem_module = dspy.Predict(GenerateProblem)
        self.fix_domain_module = dspy.Predict(FixDomainSyntax)
        self.fix_problem_module = dspy.Predict(FixProblemSyntax)
        self.fix_unsolvable_module = dspy.Predict(FixUnsolvablePDDL)
        self.fix_translate_module = dspy.Predict(FixTranslateError)

        if optimized_program is not None:
            self.load(optimized_program)

    def fix_domain(
        self,
        domain_file: Path,
        num_tries: int = 5,
    ):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_domain(domain_file)
            self.vars.domain_syntax_errors_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations domain syntax fixes: {i}")
                domain_content = self._read_pddl_file(domain_file)
                errors = err_info.get_lines_with_errors()

                self.vars.num_model_calls += 1
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
    ):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            self.vars.problem_syntax_mistakes_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations problem syntax fixes: {i}")
                domain_content = self._read_pddl_file(domain_file)
                problem_content = self._read_pddl_file(problem_file)
                errors = err_info.get_lines_with_errors()

                self.vars.num_model_calls += 1
                result = self.fix_problem_module(
                    domain_pddl=domain_content,
                    problem_pddl=problem_content,
                    errors=errors,
                )
                problem_file = self._write_pddl_file(
                    result.fixed_problem_pddl, file=problem_file
                )

    def fix_plan_not_found(
        self,
        domain_file: Path,
        problem_file: Path,
    ):
        domain_content = self._read_pddl_file(domain_file)
        problem_content = self._read_pddl_file(problem_file)

        self.vars.num_model_calls += 1
        domain_result = self.fix_unsolvable_module(
            file_to_fix=PDDLFiles.DOMAIN,
            domain_pddl=domain_content,
            problem_pddl=problem_content,
        )
        self._write_pddl_file(domain_result.fixed_pddl, file=domain_file)

        domain_content = self._read_pddl_file(domain_file)
        self.vars.num_model_calls += 1
        problem_result = self.fix_unsolvable_module(
            file_to_fix=PDDLFiles.PROBLEM,
            domain_pddl=domain_content,
            problem_pddl=problem_content,
        )
        self._write_pddl_file(problem_result.fixed_pddl, file=problem_file)

    def fix_parsing_error(
        self,
        domain_file: Path,
        problem_file: Path,
        planner_output: FDErrorInfo,
    ):
        if planner_output.file == PDDLFiles.DOMAIN:
            content = self._read_pddl_file(domain_file)
        else:
            content = self._read_pddl_file(problem_file)

        self.vars.num_model_calls += 1
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
            self.vars.generate_plan_calls += 1
            if isinstance(planner_output, Path):
                break
            elif type(planner_output) is FDErrorInfo and planner_output.exit_code in [
                ExitCodes.TRANSLATE_CRITICAL_ERROR,
                ExitCodes.TRANSLATE_INPUT_ERROR,
            ]:
                logger.debug("Parsing error")
                # NOTE: This can be triggered in some cases, where we did not find out which file
                #       the error refers to
                assert planner_output.file is not None, planner_output.error_message
                iterations = i + 1
                self.fix_parsing_error(domain_file, problem_file, planner_output)
            else:
                logger.debug("Planning error")
                iterations = i + 1
                self.fix_plan_not_found(domain_file, problem_file)
            logger.debug(f"Iterations planning fixes: {iterations}")
        assert planner_output is not None
        return planner_output

    def forward(
        self, domain_description: str, problem_description: str
    ) -> dspy.Prediction:
        self.vars.num_model_calls += 1
        domain_result = self.generate_domain_module(
            domain_description=domain_description
        )
        domain = domain_result.domain_pddl
        self.print_and_clear_history()

        domain_file = self._write_pddl_file(domain, pddl_file_type=PDDLFiles.DOMAIN)
        self.fix_domain(domain_file)
        if not is_domain_valid(domain_file):
            self.print_and_clear_history()
            logger.debug("Domain failure")
            return dspy.Prediction(out=PipelineError.DOMAIN_FAILURE, plan_file=None)
        self.print_and_clear_history()

        self.vars.num_model_calls += 1
        problem_result = self.generate_problem_module(
            domain_pddl=domain, problem_description=problem_description
        )
        problem = problem_result.problem_pddl
        self.print_and_clear_history()

        problem_file = self._write_pddl_file(problem, pddl_file_type=PDDLFiles.PROBLEM)
        self.fix_problem(domain_file, problem_file)
        if not is_problem_valid(domain_file, problem_file):
            self.print_and_clear_history()
            logger.debug("Problem failure")
            return dspy.Prediction(out=PipelineError.PROBLEM_FAILURE, plan_file=None)
        self.print_and_clear_history()

        planner_output = self.fix_planning(domain_file, problem_file)
        if isinstance(planner_output, FDErrorInfo):
            self.print_and_clear_history()
            logger.debug("Failed to generate a plan")
            return dspy.Prediction(
                out=planner_output.to_pipeline_error(), plan_file=None
            )
        logger.debug("# Successfully generated a plan")
        self.vars.plan_file = planner_output
        self.print_and_clear_history()
        return dspy.Prediction(out=None, plan_file=planner_output)

    def compile_module(self):
        return self._compile_module(True)

    def _run_impl(self):
        prediction = self(
            get_prompt(Prompts.RING_AND_PEG_DOMAIN),
            get_prompt(Prompts.RING_AND_PEG_PROBLEM),
        )
        return self.create_result(error=prediction.out)
