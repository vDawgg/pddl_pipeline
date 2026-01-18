import logging
from pathlib import Path

from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo
from src.eval.fast_downward import generate_plan as _generate_plan
from src.eval.val import get_syntax_mistakes_domain as _get_syntax_mistakes_domain
from src.eval.val import get_syntax_mistakes_problem as _get_syntax_mistakes_problem
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.utils.io import write_pddl_file
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class ToolCallPipeline(ValAndPlannerFeedbackPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(model, domain, pipeline)
        self.name = "tool_call_pipeline"
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0
        self.get_syntax_mistakes_domain_calls = 0
        self.get_syntax_mistakes_problem_calls = 0
        self.generate_plan_calls = 0
        self.tool_calls = [
            self.create_pddl_file_calls,
            self.read_pddl_file_calls,
            self.edit_lines_calls,
            self.get_syntax_mistakes_domain_calls,
            self.get_syntax_mistakes_problem_calls,
            self.generate_plan_calls,
        ]

    def get_total_tool_calls(self) -> int:
        return sum(c or 0 for c in self.tool_calls)

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
        file_path = write_pddl_file(
            content, name=self.name, pddl_file_type=pddl_file_type
        )
        if pddl_file_type == PDDLFiles.DOMAIN:
            self.domain_file = file_path
        elif pddl_file_type == PDDLFiles.PROBLEM:
            self.problem_file = file_path
        return file_path._str

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
                return "".join(f.readlines()[start : end + 1])
            return f.read()

    def edit_lines(self, file: str, line_range: tuple[int, int], replacement: str):
        """
        Replace the lines of a file in a given range with the specified replacement string.

        :param file: The file to edit the specified lines in.
        :type file: str
        :param line_range: Lines (from, to). Everything in this range will be deleted and replaced with the string specified in replacement. Note, that lines start from 0.
        :type line_range: tuple[int, int]
        :param replacement: The replacement string.
        :type replacement: str
        """
        self.edit_lines_calls += 1
        with open(file, "r+") as f:
            start, end = line_range
            lines = f.readlines()
            lines[start : end + 1] = [r + "\n" for r in replacement.split("\n")]
            f.seek(0)
            f.truncate()
            f.write("".join(lines))

    def get_syntax_mistakes_domain(self, domain_file: str) -> str:
        """
        Check the provided PDDL domain file for syntax mistakes.
        Syntax mistakes are returned with corresponding line annotations if there are any.

        :param domain_file: path to the domain file
        :type domain_file: str
        :return: possible syntax errors found in the domain file
        :rtype: str
        """
        self.get_syntax_mistakes_domain_calls += 1
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
        self.get_syntax_mistakes_problem_calls += 1
        err_info = _get_syntax_mistakes_problem(Path(domain_file), Path(problem_file))
        if err_info.num_errors > 0:
            return err_info.get_lines_with_errors()
        return "No syntax mistakes found in domain file."

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
        plan_output = _generate_plan(Path(domain_file), Path(problem_file), self.name)
        if isinstance(plan_output, FDErrorInfo):
            return plan_output.to_str()
        return f"Fast Downward successfully generated a plan under {plan_output}"

    def finish(self):
        """
        Signals that the task is complete (i.e. the desired output has been reached) and no more tools should be called.
        """
        return

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
                self.generate_plan,
                self.finish,
            ],
        )
        error = None
        if self.domain_file is None or not self.is_domain_valid(self.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.problem_file is None or not self.is_problem_valid(
            self.domain_file, self.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = _generate_plan(self.domain_file, self.problem_file, self.name)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = PipelineError.PLAN_FAILURE
            else:
                self.plan_file = plan
        return self.create_result(
            error=error,
            _number_of_fixes=self.get_total_tool_calls(),
        )
