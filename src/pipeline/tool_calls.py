import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo
from src.eval.fast_downward import translate_pddl as _translate_pddl
from src.eval.val import get_syntax_mistakes_domain as _get_syntax_mistakes_domain
from src.eval.val import get_syntax_mistakes_problem as _get_syntax_mistakes_problem
from src.inference.model_comm import (
    make_prompt_with_trajectory,
    make_tool,
    make_user_message,
    make_user_message_with_image,
    parse_react_message,
)
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.utils.prompts import Prompts, add_line_numbers, get_prompt

logger = logging.getLogger(__name__)


class ToolCallPipeline(ValAndPlannerFeedbackPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(model, domain, pipeline or Pipelines.TOOL_CALL)

    # Tools
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
        self.create_pddl_file_calls += 1
        if (self.domain_file is None and pddl_file_type == PDDLFiles.DOMAIN) or (
            self.problem_file is None and pddl_file_type == PDDLFiles.PROBLEM
        ):
            file_path = self._write_pddl_file(content, pddl_file_type=pddl_file_type)
            if pddl_file_type == PDDLFiles.DOMAIN:
                self.domain_file = file_path
            elif pddl_file_type == PDDLFiles.PROBLEM:
                self.problem_file = file_path
            return file_path._str
        # Already created a file. Preferring to use same file name to avoid model confusing
        # different versions of domain/problem file.
        assert self.domain_file is not None
        assert self.problem_file is not None
        file_path = (
            self.domain_file
            if pddl_file_type == PDDLFiles.DOMAIN
            else self.problem_file
        )
        with open(file_path, "w") as f:
            f.write(content)
            return file_path._str

    # TODO: We might need a file viewer instead
    def read_pddl_file(
        self, file: str, line_range: tuple[int, int] | None = None
    ) -> str:
        """
        Read out the content of a specified PDDL file.
        Optionally read out a subset of the file by specifying the line range.

        :param file: The file path
        :type file: str
        :param line_range: Lines (from, to). Can be specified to just fetch a subset of the given file. None by default, in which case the content of the whole file is returned.  Note, that lines start from 0.
        :type line_range: tuple[int, int] | None
        :return: The file contents
        :rtype: str
        """
        self.read_pddl_file_calls += 1
        with open(file) as f:
            if line_range is not None:
                start, end = line_range
                return "".join(add_line_numbers(f.readlines()[start : end + 1]))
            return "".join(add_line_numbers(f.readlines()))

    def edit_lines(self, file: str, line_range: tuple[int, int], new: str) -> str:
        """
        Replace the lines of a file in a given range with the specified replacement string.

        :param file: The file to edit the specified lines in.
        :type file: str
        :param line_range: Lines (from, to). Everything in this range will be deleted and replaced with the string specified in replacement. Note, that lines start from 0. Everything before the starting line and after the ending line in the original file will still persist.
        :type line_range: tuple[int, int]
        :param new: The new string.
        :type new: str
        :return: A snippet showing the updated code
        :rtype: str
        """
        # We first write the updates to a tempfile and only copy the contents once we know that no
        # additional syntax mistakes were found by VAL
        self.edit_lines_calls += 1
        start, end = line_range
        with open(file) as f:
            file_contents = f.readlines()
        temp_file = NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w") as f:
            file_contents[start : end + 1] = [r + "\n" for r in new.split("\n")]
            f.write("".join(file_contents))
        # TODO: Think about providing the would be content after the change in the message here as well
        if "domain" in file:
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

    def get_syntax_mistakes_domain(self, domain_file: str) -> str:
        """
        Check the provided PDDL domain file for syntax mistakes.
        Syntax mistakes are returned with corresponding line annotations if there are any.

        :param domain_file: path to the domain file
        :type domain_file: str
        :return: possible syntax errors found in the domain file
        :rtype: str
        """
        self.domain_syntax_errors_calls += 1
        err_info = _get_syntax_mistakes_domain(Path(domain_file))
        if err_info.num_errors > 0:
            return err_info.get_lines_with_errors()
        return "No syntax mistakes found in domain file."

    def get_syntax_mistakes_problem(self, domain_file: str, problem_file: str) -> str:
        """
        Check the provided problem file for syntax mistakes.
        Note, that the domain file the problem is based on needs to be supplied as well.
        Syntax mistakes are returned with corresponding line annotations.

        :param domain_file: path to the domain file
        :type domain_file: str
        :param problem_file: path to the problem file
        :type problem_file: str
        :return: possible syntax errors found in the problem file
        :rtype: str
        """
        # FIXME: In some cases this does not return a result (i.e. only an empty string)
        self.problem_syntax_mistakes_calls += 1
        err_info = _get_syntax_mistakes_problem(Path(domain_file), Path(problem_file))
        if err_info.num_errors > 0:
            return err_info.get_lines_with_errors()
        return "No syntax mistakes found in problem file."

    def translate_pddl(self, domain_file: str, problem_file: str) -> str:
        """
        Runs the Fast Downward translation layer. This is needed prior to generating a plan.
        If there are still syntax or semantic mistakes in the given PDDL files, the function
        return the error information.

        :param domain_file: the path to the domain file
        :type domain_file: str
        :param problem_file: the path to the problem file
        :type problem_file: str
        :return: Information on the success of the translation
        :rtype: str
        """
        self.translate_pddl_calls += 1
        translate_output = _translate_pddl(Path(domain_file), Path(problem_file))
        if translate_output is not None:
            return "Translation was unsuccessful\n" + translate_output.to_str()
        return "Fast Downward successfully translated the PDDL files for planning."

    def generate_plan(self, domain_file: str, problem_file: str) -> str:
        """
        Try to generate a plan given a domain and problem file using the fast downward planning system.
        The function either answers witht the path to the generated plan if successfull or with error information if planning was unsuccessfull.
        The main types of errors that happen either stem from the plan / domain containing syntax mistakes or the domain / problem being ill-defined leading to the planner not being able to find a plan.

        :param domain_file: the path to the domain file
        :type domain_file: str
        :param problem_file: the path to the problem file
        :type problem_file: str
        :return: The path to the problem file or an error description
        :rtype: str
        """
        self.generate_plan_calls += 1
        plan_output = self._generate_plan(Path(domain_file), Path(problem_file))
        if isinstance(plan_output, FDErrorInfo):
            return "Fast Downward was unable to generate a plan" + plan_output.to_str()
        return f"Fast Downward successfully generated a plan under {plan_output}"

    def finish(self):
        """
        Signals that the task is complete (i.e. the desired output has been reached) and no more tools should be called.
        """
        return

    # React Loop
    def make_react_workflow(
        self,
        input_prompt: str,
        tools: list[Callable],
        image_paths: list[str] | None = None,
        max_iters: int = 10,
        max_past_setup: int = 3,
    ) -> str:
        tools_json = [make_tool(t) for t in tools]
        tools_dict = {t.__name__: t for t in tools}

        unformatted_base_prompt = get_prompt(Prompts.REACT_BASE)
        base_prompt = unformatted_base_prompt.format(tools=tools_json)
        input_prompt = base_prompt + input_prompt

        res = None
        parsed_responses = []
        tool_results = []
        past_tool_name = None
        past_tool_args = None
        for _ in range(max_iters):
            prompt_with_trajectory = make_prompt_with_trajectory(
                input_prompt,
                parsed_responses,
                tool_results,
                domain_file_path=self.domain_file,
                problem_file_path=self.problem_file,
                max_past_steps=max_past_setup,
            )
            logger.debug(f"# Prompts with trajectory:\n{prompt_with_trajectory}")
            res = (
                self._openai_client.chat.completions.create(
                    model=self._model_config.api_model_name,
                    messages=[
                        make_user_message(prompt_with_trajectory)
                        if image_paths is None
                        else make_user_message_with_image(
                            prompt_with_trajectory, image_paths
                        )  # type: ignore
                    ],
                )
                .choices[0]
                .message.content
            )
            self.num_model_calls += 1
            assert res is not None
            logger.debug("# Assistant Message")
            logger.debug(res)

            parsed_response = parse_react_message(res)
            if parsed_response is None:
                logger.debug("Assistant answered with malformed response")
                # TODO: Think about what we actually want to do here
                # TODO: We should also log these as results
                continue

            parsed_responses.append(parsed_response)
            if parsed_response.tool_name == "finish":
                break
            if (
                parsed_response.tool_name == past_tool_name
                and parsed_response.tool_args == past_tool_args
            ):
                tool_results.append(
                    "Tool call is repetition of previous call. Tool was not called. Please do not repeat your actions!"
                )
            else:
                try:
                    tool_results.append(
                        tools_dict[parsed_response.tool_name](
                            **parsed_response.tool_args
                        )
                    )
                except Exception as e:
                    logger.debug("Assistant answered with unusable tool-call")
                    logger.debug(f"Error message: {e}")
                    parsed_responses.pop()
                    continue
            logger.debug("# Tool Call")
            logger.debug(f"## Tool: {parsed_response.tool_name}")
            logger.debug(f"## Tool args: {parsed_response.tool_args}")
            logger.debug(f"## Tool result: {tool_results[-1]}")
            past_tool_name = parsed_response.tool_name
            past_tool_args = parsed_response.tool_args
        assert res is not None
        return res.strip()

    def _run_impl(self) -> PipelineResult:
        self.make_react_workflow(
            input_prompt=get_prompt(
                Prompts.GENERATION_CONTEXT_TOOLS, Prompts.RING_AND_PEG
            ),
            tools=[
                self.create_pddl_file,
                self.read_pddl_file,
                self.edit_lines,
                self.get_syntax_mistakes_domain,
                self.get_syntax_mistakes_problem,
                self.translate_pddl,
                self.generate_plan,
                self.finish,
            ],
            max_iters=20,
        )
        error = None
        if self.domain_file is None or not self.is_domain_valid(self.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.problem_file is None or not self.is_problem_valid(
            self.domain_file, self.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = self._generate_plan(self.domain_file, self.problem_file)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = plan.to_pipeline_error()
            else:
                self.plan_file = plan
        return self.create_result(
            error=error,
        )
