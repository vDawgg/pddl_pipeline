import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import dspy

from src.base.pipeline import FDErrorInfo, PipelineBase
from src.base.schema import PDDLFiles, PipelineError, Pipelines, Tools
from src.eval.fast_downward import translate_pddl as _translate_pddl
from src.eval.val import get_syntax_mistakes_domain as _get_syntax_mistakes_domain
from src.eval.val import get_syntax_mistakes_problem as _get_syntax_mistakes_problem
from src.eval.val import is_domain_valid, is_problem_valid
from src.inference import Models
from src.utils.domains import Domains
from src.utils.prompts import Prompts, add_line_numbers, get_prompt

logger = logging.getLogger(__name__)


class GeneratePddlSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_TOOLS)

    task_description: str = dspy.InputField(
        desc="Natural language description of the planning task"
    )
    success: Literal["success", "failure"] = dspy.OutputField(
        desc="Whether the final PDDL domain and problem are syntactically valid and solvable."
    )


# NOTE: This would likely benefit from its own separate optimization but is likely going to be to costly
class PlanFeedbackSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_PLAN_FEEDBACK)

    task_and_plan: str = dspy.InputField(
        desc="The generated plan and a natural language description of the task the plan should be applied to."
    )
    feedback: str = dspy.OutputField(desc="Actionable feedback on the provided plan.")


class ToolCallPipeline(PipelineBase):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        super().__init__(
            model,
            domain,
            ablate_tools,
            pipeline or Pipelines.TOOL_CALL,
            optimized_program,
        )
        self.tools = {
            Tools.CREATE_PDDL_FILE: self.create_pddl_file,
            Tools.READ_PDDL_FILE: self.read_pddl_file,
            Tools.EDIT_LINES: self.edit_lines,
            Tools.GET_SYNTAX_MISTAKES_DOMAIN: self.get_syntax_mistakes_domain,
            Tools.GET_SYNTAX_MISTAKES_PROBLEM: self.get_syntax_mistakes_problem,
            Tools.TRANSLATE_PDDL: self.translate_pddl,
            Tools.GENERATE_PLAN: self.generate_plan,
            Tools.GET_PLAN_FEEDBACK: self.get_plan_feedback,
        }
        # Remove tools for ablation
        if ablate_tools is not None:
            for tool in ablate_tools:
                if tool in self.tools:
                    del self.tools[tool]
                elif tool is not None:
                    logger.info(f"{tool} could not be ablated, as it is not available.")
        logger.debug(f"Available tools: {list(self.tools.keys())}")
        self.generate_pddl_module = dspy.ReAct(
            GeneratePddlSignature,
            tools=list(self.tools.values()),
            max_iters=30,
        )
        self.plan_feedback_module = dspy.Predict(PlanFeedbackSignature)
        if optimized_program is not None:
            self.load(optimized_program)

    ## TOOLS

    def create_pddl_file(self, content: str, pddl_file_type: PDDLFiles) -> str:
        """
        Creates a pddl file with the given content.
        The filename and path are set automatically and returned by this function.

        :param content: The PDDL file content.
        :type content: str
        :param pddl_file_type: The type of the PDDL file (either 'domain' or 'problem').
        :type pddl_file_type: PDDLFiles
        :return: The path of the newly created file.
        :rtype: str
        """
        self.vars.create_pddl_file_calls += 1
        if (self.vars.domain_file is None and pddl_file_type == PDDLFiles.DOMAIN) or (
            self.vars.problem_file is None and pddl_file_type == PDDLFiles.PROBLEM
        ):
            file_path = self._write_pddl_file(content, pddl_file_type=pddl_file_type)
            if pddl_file_type == PDDLFiles.DOMAIN:
                self.vars.domain_file = file_path
            elif pddl_file_type == PDDLFiles.PROBLEM:
                self.vars.problem_file = file_path
            return file_path._str
        # Already created a file. Preferring to use same file name to avoid model confusing
        # different versions of domain/problem file.
        file_path = (
            self.vars.domain_file
            if pddl_file_type == PDDLFiles.DOMAIN
            else self.vars.problem_file
        )
        assert file_path is not None
        with open(file_path, "w") as f:
            f.write(content)
            return "File was created successfully"

    def read_pddl_file(
        self, pddl_file_type: PDDLFiles, line_range: tuple[int, int] | None = None
    ) -> str:
        """
        Read out the content of a specified PDDL file.
        Optionally read out a subset of the file by specifying the line range.

        :param pddl_file_type: The type of the PDDL file (either 'domain' or 'problem').
        :param line_range: Lines (from, to). Can be specified to just fetch a subset of the given file. None by default, in which case the content of the whole file is returned.  Note, that lines start from 0.
        :type line_range: tuple[int, int] | None
        :return: The file contents
        :rtype: str
        """
        if pddl_file_type == PDDLFiles.DOMAIN:
            if self.vars.domain_file is None:
                return "Domain file has not yet been created"
            file = self.vars.domain_file
        elif pddl_file_type == PDDLFiles.PROBLEM:
            if self.vars.problem_file is None:
                return "Problem file has not yet been created"
            file = self.vars.problem_file
        self.vars.read_pddl_file_calls += 1
        with open(file) as f:
            if line_range is not None:
                start, end = line_range
                return "".join(add_line_numbers(f.readlines()[start : end + 1]))
            return "".join(add_line_numbers(f.readlines()))

    def edit_lines(
        self, pddl_file_type: PDDLFiles, line_range: tuple[int, int], new: str
    ) -> str:
        """
        Replace the lines of a file in a given range with the specified replacement string.

        :param pddl_file_type: The type of the PDDL file (either 'domain' or 'problem').
        :param line_range: Lines (from, to). Everything in this range will be deleted and replaced with the string specified in replacement. Note, that lines start from 0. Everything before the starting line and after the ending line in the original file will still persist.
        :type line_range: tuple[int, int]
        :param new: The new string.
        :type new: str
        :return: A snippet showing the updated code
        :rtype: str
        """
        # We first write the updates to a tempfile and only copy the contents once we know that no
        # additional syntax mistakes were found by VAL
        self.vars.edit_lines_calls += 1
        start, end = line_range
        if pddl_file_type == PDDLFiles.DOMAIN:
            if self.vars.domain_file is None:
                return "Domain file has not yet been created"
            file = self.vars.domain_file
        elif pddl_file_type == PDDLFiles.PROBLEM:
            if self.vars.problem_file is None:
                return "Problem file has not yet been created"
            file = self.vars.problem_file
        with open(file) as f:
            file_contents = f.readlines()
        temp_file = NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w") as f:
            file_contents[start : end + 1] = [r + "\n" for r in new.split("\n")]
            f.write("".join(file_contents))
        # TODO: Think about providing the would be content after the change in the message here as well
        if "domain" in str(file):
            err_info = _get_syntax_mistakes_domain(Path(temp_file.name))
            logger.debug(f"Error Info domain after edit: {err_info.errors}")
            if err_info.num_errors > 0:
                return (
                    "Found syntax errors! Your edit was not applied to the file!\n"
                    + err_info.get_lines_with_errors()
                )
        else:
            err_info = _get_syntax_mistakes_problem(
                Path(self.domain), Path(temp_file.name)
            )
            logger.debug(f"Error Info problem after edit: {err_info.errors}")
            if err_info.num_errors > 0:
                return (
                    "Found syntax errors! Your edit was not applied to the file!\n"
                    + f"{err_info.get_lines_with_errors()}\n"
                )
        shutil.copy(temp_file.name, file)
        content = "".join(add_line_numbers(file_contents))
        return f"**Edit successfully applied.**\n\n## Updated file contents:\n\n```pddl\n{content}\n```"

    def get_syntax_mistakes_domain(self) -> str:
        """
        Check the generated PDDL domain file for syntax mistakes.
        Syntax mistakes are returned with corresponding line annotations if there are any.

        :return: possible syntax errors found in the domain file
        :rtype: str
        """
        self.vars.domain_syntax_errors_calls += 1
        if self.vars.domain_file is None:
            return "Domain file has not yet been created"
        err_info = _get_syntax_mistakes_domain(Path(self.vars.domain_file))
        if err_info.num_errors > 0:
            return err_info.get_lines_with_errors()
        return "No syntax mistakes found in domain file."

    def get_syntax_mistakes_problem(self) -> str:
        """
        Check the generated problem file for syntax mistakes.
        Syntax mistakes are returned with corresponding line annotations.

        :return: possible syntax errors found in the problem file
        :rtype: str
        """
        self.vars.problem_syntax_mistakes_calls += 1
        if self.vars.domain_file is None:
            return "Domain file has not yet been created"
        elif self.vars.problem_file is None:
            return "Problem file has not yet been created"
        err_info = _get_syntax_mistakes_problem(
            Path(self.vars.domain_file), Path(self.vars.problem_file)
        )
        if err_info.num_errors > 0:
            return err_info.get_lines_with_errors()
        return "No syntax mistakes found in problem file."

    def translate_pddl(self) -> str:
        """
        Runs the Fast Downward translation layer. This is needed prior to generating a plan.
        If there are still syntax or semantic mistakes in the PDDL files, the function
        return the error information.

        :return: Information on the success of the translation
        :rtype: str
        """
        self.vars.translate_pddl_calls += 1
        if self.vars.domain_file is None:
            return "Domain file has not yet been created"
        elif self.vars.problem_file is None:
            return "Problem file has not yet been created"
        translate_output = _translate_pddl(
            Path(self.vars.domain_file), Path(self.vars.problem_file)
        )
        if translate_output is not None:
            return "Translation was unsuccessful\n" + translate_output.to_str()
        return "Fast Downward successfully translated the PDDL files for planning."

    def generate_plan(self) -> str:
        """
        Try to generate a plan given a domain and problem file using the fast downward planning system.
        The function answers with the generated plan or error information if the generation fails.
        The main types of errors that happen either stem from the plan / domain containing syntax mistakes or the domain / problem being ill-defined leading to the planner not being able to find a plan.

        :return: The outcome of the planning process
        :rtype: str
        """
        self.vars.generate_plan_calls += 1
        if self.vars.domain_file is None:
            return "Domain file has not yet been created"
        elif self.vars.problem_file is None:
            return "Problem file has not yet been created"
        plan_output = self._generate_plan(
            Path(self.vars.domain_file), Path(self.vars.problem_file)
        )
        if isinstance(plan_output, FDErrorInfo):
            return (
                "Fast Downward was unable to generate a plan\n" + plan_output.to_str()
            )
        self.vars.plan_file = plan_output
        with open(plan_output) as f:
            return f"Fast Downward successfully generated a plan.\n\nGenerated plan:\n{f.read()}"

    def get_plan_feedback(self) -> str:
        """
        Get detailed feedback on the feasibility of a generated plan given the current task.

        :return: Feedback on the physical/logical feasibility of the generated plan.
        :rtype: str
        """
        self.vars.get_plan_feedback_calls += 1
        base_plan_feedback_prompt = get_prompt(Prompts.PLAN_FEEDBACK)
        if self.vars.plan_file is None:
            return "Could not generate feedback, as no plan has been generated to provide feedback on."
        with open(self.vars.plan_file) as f:
            plan = f.read()
        plan_feedback_prompt = base_plan_feedback_prompt.format(
            # TODO: We need to update this to accept/gather other domain/problem instructions than ring_and_peg in the future
            task=get_prompt(Prompts.RING_AND_PEG),
            plan=plan,
        )
        return self.plan_feedback_module(task_and_plan=plan_feedback_prompt).feedback

    def forward(
        self, task_description: str, scene: dspy.Image | None = None
    ) -> dspy.Prediction:
        error = None
        self.generate_pddl_module(
            task_description=task_description,
        )
        if self.vars.domain_file is None or not is_domain_valid(self.vars.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.vars.problem_file is None or not is_problem_valid(
            self.vars.domain_file, self.vars.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = self._generate_plan(self.vars.domain_file, self.vars.problem_file)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = plan.to_pipeline_error()
            else:
                self.vars.plan_file = plan
                return dspy.Prediction(out=error, plan_file=plan)
        return dspy.Prediction(out=error, plan_file=None)

    ## MODULE OPTIMIZATION FUNCTIONS

    def compile_module(self):
        return self._compile_module()

    ## PIPELINE EXECUTION

    def _run_impl(self):
        prediction = self(get_prompt(Prompts.RING_AND_PEG))
        token_usage = prediction.get_lm_usage()
        assert token_usage is not None, "token_usage is None"
        token_usage = token_usage[f"openai/{self._model_config.api_model_name}"]
        self.vars.input_tokens = token_usage["prompt_tokens"]
        self.vars.output_tokens = token_usage["completion_tokens"]
        self.log_and_clear_history()
        return self.create_result(error=prediction.out)
