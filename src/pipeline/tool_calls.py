from pathlib import Path

from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo, generate_plan
from src.inference.model_comm import make_react_workflow
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.utils.io import read_pddl_file, write_pddl_file
from src.utils.prompts import Prompts, get_prompt


class ToolCallPipeline(ValAndPlannerFeedbackPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(model, domain, pipeline)
        self.domain_file: Path | None = None
        self.problem_file: Path | None = None
        self.create_pddl_file_calls = 0
        self.read_pddl_file_calls = 0
        self.edit_lines_calls = 0

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
        :param line_range: Lines (from, to). Can be specified to just fetch a subset of the given file. None by default, in which case the content of the whole file is returned.
        :type line_range: tuple[int, int] | None
        :return: The file contents
        :rtype: str
        """
        self.read_pddl_file_calls += 1
        if line_range is not None:
            start, end = line_range
            return read_pddl_file(Path(file))[start:end]
        return read_pddl_file(Path(file))

    def edit_lines(self, file: str, line_range: tuple[int, int], replacement: str):
        """
        Replace the lines of a file in a given range with the specified replacement string.

        :param file: The file to edit the specified lines in.
        :type file: str
        :param line_range: Lines (from, to). Everything in this range will be deleted and replaced with the string specified in replacement.
        :type line_range: tuple[int, int]
        :param replacement: The replacement string.
        :type replacement: str
        """
        pass

    def finish(self):
        """
        Signals that the task is complete (i.e. the desired output has been reached) and no more tools should be called.
        """
        return

    def run(self):
        make_react_workflow(
            model_name=self.model,
            input_prompt=get_prompt(Prompts.GENERATION_CONTEXT, Prompts.RING_AND_PEG),
            tools=[self.create_pddl_file, self.read_pddl_file, self.finish],
        )
        # TODO: We need to make this a unified definition. This discrepancy between the tool call and the other pipelines
        #       is not desirable.
        #       -> The calls below could possibly just be gathered into one file-editing count / bucket
        print("# Create PDDL file calls:", self.create_pddl_file_calls)
        print("# Read PDDL file calls:", self.read_pddl_file_calls)
        print("# Edit lines calls:", self.edit_lines_calls)
        error = None
        plan = None
        if self.domain_file is None or not self.is_domain_valid(self.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.problem_file is None or not self.is_problem_valid(
            self.domain_file, self.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = generate_plan(self.domain_file, self.problem_file, self.name)
            if isinstance(plan, FDErrorInfo):
                error = PipelineError.PLAN_FAILURE
                plan = None
        return PipelineResult(
            error=error,
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            plan_file=plan,
            _number_of_fixes=sum(
                c or 0
                for c in [
                    self.create_pddl_file_calls,
                    self.read_pddl_file_calls,
                    self.edit_lines_calls,
                ]
            ),
        )
