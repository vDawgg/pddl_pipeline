import logging
from pathlib import Path

import dspy

from src.base.pipeline import PipelineBase
from src.base.schema import PDDLFiles, PipelineError, Pipelines
from src.constants import images_dir
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


class GenerateDomainImage(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_IMAGES)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    domain_description: str = dspy.InputField(
        desc="Natural language description of the planning domain to model"
    )
    domain_pddl: str = dspy.OutputField(
        desc="Complete PDDL domain file adhering to PDDL 1.0 standard"
    )


class GenerateProblemImage(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_IMAGES)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    domain_pddl: str = dspy.InputField(desc="The previously generated PDDL domain file")
    problem_description: str = dspy.InputField(
        desc="Natural language description of the specific problem instance"
    )
    problem_pddl: str = dspy.OutputField(
        desc="Complete PDDL problem file adhering to PDDL 1.0 standard"
    )


class FixDomainSyntaxImage(dspy.Signature):
    __doc__ = get_prompt(
        Prompts.VAL_FEEDBACK_CONTEXT_IMAGES, Prompts.VAL_FEEDBACK_DOMAIN
    )

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    domain_pddl: str = dspy.InputField(
        desc="The PDDL domain file containing syntax errors"
    )
    errors: str = dspy.InputField(
        desc="List of syntax errors with line numbers and error messages"
    )
    fixed_domain_pddl: str = dspy.OutputField(
        desc="The corrected PDDL domain file with all syntax errors fixed"
    )


class FixProblemSyntaxImage(dspy.Signature):
    __doc__ = get_prompt(
        Prompts.VAL_FEEDBACK_CONTEXT_IMAGES, Prompts.VAL_FEEDBACK_PROBLEM
    )

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
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


class FixUnsolvablePDDLImage(dspy.Signature):
    __doc__ = get_prompt(Prompts.PLANNER_CONTEXT_IMAGES, Prompts.PLANNER_TASK)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    file_to_fix: str = dspy.InputField(desc="Which file to fix: 'domain' or 'problem'")
    domain_pddl: str = dspy.InputField(desc="The current PDDL domain file")
    problem_pddl: str = dspy.InputField(desc="The current PDDL problem file")
    fixed_pddl: str = dspy.OutputField(
        desc="The fixed PDDL file (domain or problem as specified)"
    )


class FixTranslateErrorImage(dspy.Signature):
    __doc__ = get_prompt(
        Prompts.PLANNER_TRANSLATE_CONTEXT_IMAGES, Prompts.PLANNER_TRANSLATE_TASK
    )

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    file_to_fix: str = dspy.InputField(desc="Which file to fix: 'domain' or 'problem'")
    error_message: str = dspy.InputField(
        desc="The error message from the planner's translator"
    )
    file_content: str = dspy.InputField(desc="The content of the file to fix")
    fixed_pddl: str = dspy.OutputField(
        desc="The fixed PDDL file with the translation error resolved"
    )


class RigidTrajectoryImagePipeline(PipelineBase):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(
            model, domain, pipeline=pipeline or Pipelines.RIGID_TRAJECTORY_IMAGE
        )

        self.generate_domain_module_image = dspy.Predict(GenerateDomainImage)
        self.generate_problem_module_image = dspy.Predict(GenerateProblemImage)
        self.fix_domain_module_image = dspy.Predict(FixDomainSyntaxImage)
        self.fix_problem_module_image = dspy.Predict(FixProblemSyntaxImage)
        self.fix_unsolvable_module_image = dspy.Predict(FixUnsolvablePDDLImage)
        self.fix_translate_module_image = dspy.Predict(FixTranslateErrorImage)

    def fix_domain(
        self,
        domain_file: Path,
        num_tries: int = 5,
        image: dspy.Image | None = None,
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
                result = self.fix_domain_module_image(
                    scene=image, domain_pddl=domain_content, errors=errors
                )
                domain_file = self._write_pddl_file(
                    result.fixed_domain_pddl, file=domain_file
                )

    def fix_problem(
        self,
        domain_file: Path,
        problem_file: Path,
        num_tries: int = 5,
        image: dspy.Image | None = None,
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
                result = self.fix_problem_module_image(
                    scene=image,
                    domain_pddl=domain_content,
                    problem_pddl=problem_content,
                    errors=errors,
                )
                problem_file = self._write_pddl_file(
                    result.fixed_problem_pddl, file=problem_file
                )

    def fix_plan_not_found(
        self, domain_file: Path, problem_file: Path, image: dspy.Image | None = None
    ):
        domain_content = self._read_pddl_file(domain_file)
        problem_content = self._read_pddl_file(problem_file)

        self.vars.num_model_calls += 1
        domain_result = self.fix_unsolvable_module_image(
            scene=image,
            file_to_fix=PDDLFiles.DOMAIN,
            domain_pddl=domain_content,
            problem_pddl=problem_content,
        )
        self._write_pddl_file(domain_result.fixed_pddl, file=domain_file)

        domain_content = self._read_pddl_file(domain_file)
        self.vars.num_model_calls += 1
        problem_result = self.fix_unsolvable_module_image(
            scene=image,
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
        image: dspy.Image | None = None,
    ):
        if planner_output.file == PDDLFiles.DOMAIN:
            content = self._read_pddl_file(domain_file)
        else:
            content = self._read_pddl_file(problem_file)

        self.vars.num_model_calls += 1
        result = self.fix_translate_module_image(
            scene=image,
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
        image: dspy.Image | None = None,
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
                self.fix_parsing_error(domain_file, problem_file, planner_output, image)
            else:
                logger.debug("Planning error")
                iterations = i + 1
                self.fix_plan_not_found(domain_file, problem_file, image)
            logger.debug(f"Iterations planning fixes: {iterations}")
        assert planner_output is not None
        return planner_output

    def forward(
        self,
        domain_description: str,
        problem_description: str,
        image_path: str,
    ) -> dspy.Prediction:
        image = dspy.Image(str(images_dir / image_path))

        self.vars.num_model_calls += 1
        domain_result = self.generate_domain_module_image(
            scene=image, domain_description=domain_description
        )
        domain = domain_result.domain_pddl
        self.log_and_clear_history()

        self.vars.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        self.fix_domain(self.vars.domain_file, image=image)
        if not is_domain_valid(self.vars.domain_file):
            self.log_and_clear_history()
            logger.debug("Domain failure")
            return dspy.Prediction(out=PipelineError.DOMAIN_FAILURE)
        self.log_and_clear_history()

        self.vars.num_model_calls += 1
        problem_result = self.generate_problem_module_image(
            scene=image, domain_pddl=domain, problem_description=problem_description
        )
        problem = problem_result.problem_pddl
        self.log_and_clear_history()

        self.vars.problem_file = self._write_pddl_file(
            problem, pddl_file_type=PDDLFiles.PROBLEM
        )
        self.fix_problem(self.vars.domain_file, self.vars.problem_file, image=image)
        if not is_problem_valid(self.vars.domain_file, self.vars.problem_file):
            self.log_and_clear_history()
            logger.debug("Problem failure")
            return dspy.Prediction(out=PipelineError.PROBLEM_FAILURE)
        self.log_and_clear_history()

        planner_output = self.fix_planning(
            self.vars.domain_file, self.vars.problem_file, image=image
        )
        if isinstance(planner_output, FDErrorInfo):
            self.log_and_clear_history()
            logger.debug("Failed to generate a plan")
            return dspy.Prediction(out=planner_output.to_pipeline_error())
        logger.debug("# Successfully generated a plan")
        self.vars.plan_file = planner_output
        self.log_and_clear_history()
        return dspy.Prediction(out=None)

    def compile_module(self):
        # TODO: Implement if needed. This will require image input to be passed in the compile functions as well.
        #       Images can be gathered from pddlgym but might be out of scope for this
        ...

    def _run_impl(self):
        prediction = self(
            get_prompt(Prompts.RING_AND_PEG_DOMAIN),
            get_prompt(Prompts.RING_AND_PEG_PROBLEM),
            "peg_and_ring_plan_start.png",
        )
        token_usage = prediction.get_lm_usage()
        assert token_usage is not None
        token_usage = token_usage[f"openai/{self._model_config.api_model_name}"]
        self.vars.input_tokens = token_usage["prompt_tokens"]
        self.vars.output_tokens = token_usage["completion_tokens"]
        return self.create_result(error=prediction.out)
