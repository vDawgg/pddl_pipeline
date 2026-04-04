import logging
from pathlib import Path

import dspy

from src.base.mappings import (
    ACTION_SCHEMA_PROMPTS,
    DOMAIN_PROMPTS,
    IMAGES,
    PROBLEM_PROMPTS,
)
from src.base.pipeline import PipelineBase
from src.base.schemas import Domains, PDDLFiles, PipelineError, Pipelines, Problems
from src.constants import images_dir
from src.eval.fast_downward import ExitCodes, FDErrorInfo
from src.eval.val import (
    get_syntax_mistakes_domain,
    get_syntax_mistakes_problem,
    is_domain_valid,
    is_problem_valid,
)
from src.inference import Models
from src.utils.prompts import Prompts, get_domain_problem_prompt, get_prompt

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
        desc="Complete PDDL domain file adhering to PDDL 1.2 standard"
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
        desc="Complete PDDL problem file adhering to PDDL 1.2 standard"
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


class PlanFeedbackImageSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_PLAN_FEEDBACK)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    task_and_plan: str = dspy.InputField(
        desc="The generated plan and a natural language description of the task the plan should be applied to."
    )
    feedback: str = dspy.OutputField(desc="Actionable feedback on the provided plan.")


class FixPlanFeedbackImage(dspy.Signature):
    __doc__ = get_prompt(Prompts.PLANNER_CONTEXT_IMAGES)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    file_to_fix: str = dspy.InputField(desc="Which file to fix: 'domain' or 'problem'")
    domain_pddl: str = dspy.InputField(desc="The current PDDL domain file")
    problem_pddl: str = dspy.InputField(desc="The current PDDL problem file")
    plan_feedback: str = dspy.InputField(
        desc="Feedback on the feasibility of the generated plan"
    )
    fixed_pddl: str = dspy.OutputField(
        desc="The fixed PDDL file (domain or problem as specified)"
    )


class ActionMappingImageSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_ACTION_MAPPING)

    plan_and_actions: str = dspy.InputField(
        desc="The generated plan, PDDL domain and schema of the robots actions."
    )
    mapped_plan: str = dspy.OutputField(
        desc="The plan with the actions mapped to the robots action schema."
    )


class RigidTrajectoryImagePipeline(PipelineBase):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        problem: Problems,
        pipeline: Pipelines | None = None,
    ):
        super().__init__(
            model,
            domain,
            problem,
            pipeline=pipeline or Pipelines.RIGID_TRAJECTORY_IMAGE,
        )

        self.generate_domain_module_image = dspy.Predict(GenerateDomainImage)
        self.generate_problem_module_image = dspy.Predict(GenerateProblemImage)
        self.fix_domain_module_image = dspy.Predict(FixDomainSyntaxImage)
        self.fix_problem_module_image = dspy.Predict(FixProblemSyntaxImage)
        self.fix_unsolvable_module_image = dspy.Predict(FixUnsolvablePDDLImage)
        self.fix_translate_module_image = dspy.Predict(FixTranslateErrorImage)
        self.plan_feedback_module_image = dspy.Predict(PlanFeedbackImageSignature)
        self.fix_plan_feedback_module_image = dspy.Predict(FixPlanFeedbackImage)
        self.action_mapping_module = dspy.Predict(ActionMappingImageSignature)

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

    def get_plan_feedback(
        self,
        plan_file: Path,
        image: dspy.Image | None = None,
    ) -> str:
        base_plan_feedback_prompt = get_prompt(Prompts.PLAN_FEEDBACK)
        with open(plan_file) as f:
            plan = f.read()
        plan_feedback_prompt = base_plan_feedback_prompt.format(
            task=get_domain_problem_prompt(
                DOMAIN_PROMPTS[self.domain],
                PROBLEM_PROMPTS[self.problem],
            ),
            plan=plan,
        )
        self.vars.get_plan_feedback_calls += 1
        return self.plan_feedback_module_image(
            scene=image, task_and_plan=plan_feedback_prompt
        ).feedback

    def incorporate_plan_feedback(
        self,
        domain_file: Path,
        problem_file: Path,
        plan_file: Path,
        num_tries: int = 3,
        image: dspy.Image | None = None,
    ) -> Path | None:
        for i in range(num_tries):
            feedback = self.get_plan_feedback(plan_file, image)
            self.log_and_clear_history()

            domain_content = self._read_pddl_file(domain_file)
            problem_content = self._read_pddl_file(problem_file)

            self.vars.num_model_calls += 1
            domain_result = self.fix_plan_feedback_module_image(
                scene=image,
                file_to_fix=PDDLFiles.DOMAIN,
                domain_pddl=domain_content,
                problem_pddl=problem_content,
                plan_feedback=feedback,
            )
            improved_domain = self._write_pddl_file(
                domain_result.fixed_pddl, pddl_file_type=PDDLFiles.DOMAIN
            )

            domain_content = self._read_pddl_file(domain_file)
            self.vars.num_model_calls += 1
            problem_result = self.fix_plan_feedback_module_image(
                scene=image,
                file_to_fix=PDDLFiles.PROBLEM,
                domain_pddl=domain_content,
                problem_pddl=problem_content,
                plan_feedback=feedback,
            )
            improved_problem = self._write_pddl_file(
                problem_result.fixed_pddl, pddl_file_type=PDDLFiles.PROBLEM
            )
            self.log_and_clear_history()

            self.fix_domain(improved_domain, image=image)
            if not is_domain_valid(improved_domain):
                logger.debug(f"Plan feedback iteration {i}: domain invalid after fix")
                continue
            self.log_and_clear_history()

            self.fix_problem(improved_domain, improved_problem, image=image)
            if not is_problem_valid(improved_domain, improved_problem):
                logger.debug(f"Plan feedback iteration {i}: problem invalid after fix")
                continue
            self.log_and_clear_history()

            planner_output = self.fix_planning(
                improved_domain, improved_problem, image=image
            )
            if isinstance(planner_output, FDErrorInfo):
                logger.debug(f"Plan feedback iteration {i}: planning failed")
                continue
            self.log_and_clear_history()

            # Incorporating feedback was successfull. Can now overwrite old domain and problem
            # with new instances
            plan_file = planner_output
            logger.debug(f"Plan feedback iteration {i}: successfully re-planned")
            self._write_pddl_file(domain_result.fixed_pddl, file=domain_file)
            self._write_pddl_file(problem_result.fixed_pddl, file=problem_file)
            return plan_file
        return None

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
        action_schema: str,
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

        # Plan feedback loop: get feedback and allow model to fix domain/problem
        plan_file = self.incorporate_plan_feedback(
            self.vars.domain_file,
            self.vars.problem_file,
            planner_output,
            image=image,
        )
        self.vars.plan_file = plan_file
        if self.vars.plan_file is None:
            logger.debug("# Failed to incorporate plan feedback")
            return dspy.Prediction(
                out=PipelineError.FEEDBACK_INCORPORATION_FAILURE, plan_file=None
            )

        # Map plan to action schema of robot
        base_action_mapping_prompt = get_prompt(Prompts.ACTION_MAPPING)
        with open(self.vars.plan_file) as f:
            action_mapping_prompt = base_action_mapping_prompt.format(
                domain=self._read_pddl_file(self.vars.domain_file),
                plan=f.read(),
                action_schema=action_schema,
            )
            plan_mapping_out = self.action_mapping_module(
                plan_and_actions=action_mapping_prompt
            )
        with open(self.vars.plan_file, "w") as f:
            f.write(plan_mapping_out.mapped_plan)
        self.log_and_clear_history()

        return dspy.Prediction(out=None, plan_file=self.vars.plan_file)

    def compile_module(self):
        # TODO: Implement if needed. This will require image input to be passed in the compile functions as well.
        #       Images can be gathered from pddlgym but might be out of scope for this
        ...

    def _run_impl(self):
        prediction = self(
            get_prompt(Prompts.DOMAIN).format(
                domain=get_prompt(DOMAIN_PROMPTS[self.domain])
            ),
            get_prompt(Prompts.PROBLEM).format(
                problem=get_prompt(PROBLEM_PROMPTS[self.problem])
            ),
            IMAGES[self.problem],
            action_schema=get_prompt(ACTION_SCHEMA_PROMPTS[self.domain]),
        )
        token_usage = prediction.get_lm_usage()
        assert token_usage is not None
        token_usage = token_usage[f"openai/{self._model_config.api_model_name}"]
        self.vars.input_tokens = token_usage["prompt_tokens"]
        self.vars.output_tokens = token_usage["completion_tokens"]
        return self.create_result(error=prediction.out)
