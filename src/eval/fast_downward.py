"""
The parsing code is partially based on the fastdownward parsers from the downward lab project
https://github.com/aibasel/lab/tree/main/downward/parsers
"""

from enum import StrEnum, auto

from subprocess import PIPE, run


class ExitCodes(StrEnum):
    # Successfull terminations
    SUCCESS = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_TIME = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME = auto()
    # Unsuccessfull but error free termination
    TRANSLATE_UNSOLVABLE = auto()
    SEARCH_UNSOLVABLE = auto()
    SEARCH_UNSOLVED_INCOMPLETE = auto()
    # Expected failures
    TRANSLATE_OUT_OF_MEMORY = auto()
    TRANSLATE_OUT_OF_TIME = auto()
    SEARCH_OUT_OF_MEMORY = auto()
    SEARCH_OUT_OF_TIME = auto()
    SEARCH_OUT_OF_MEMORY_AND_TIME = auto()
    # Unrecoverable failures
    TRANSLATE_CRITICAL_ERROR = auto()
    TRANSLATE_INPUT_ERROR = auto()
    SEARCH_CRITICAL_ERROR = auto()
    SEARCH_INPUT_ERROR = auto()
    SEARCH_UNSUPPORTED = auto()
    DRIVER_CRITICAL_ERROR = auto()
    DRIVER_INPUT_ERROR = auto()
    DRIVER_UNSUPPORTED = auto()


exit_codes = {
    0: ExitCodes.SUCCESS,
    1: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY,
    2: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME,
    3: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME,
    10: ExitCodes.TRANSLATE_UNSOLVABLE,
    11: ExitCodes.SEARCH_UNSOLVABLE,
    12: ExitCodes.SEARCH_UNSOLVED_INCOMPLETE,
    20: ExitCodes.TRANSLATE_OUT_OF_MEMORY,
    21: ExitCodes.TRANSLATE_OUT_OF_TIME,
    22: ExitCodes.SEARCH_OUT_OF_MEMORY,
    23: ExitCodes.SEARCH_OUT_OF_TIME,
    24: ExitCodes.SEARCH_OUT_OF_MEMORY_AND_TIME,
    30: ExitCodes.TRANSLATE_CRITICAL_ERROR,
    31: ExitCodes.TRANSLATE_INPUT_ERROR,
    32: ExitCodes.SEARCH_CRITICAL_ERROR,
    33: ExitCodes.SEARCH_INPUT_ERROR,
    34: ExitCodes.SEARCH_UNSUPPORTED,
    35: ExitCodes.DRIVER_CRITICAL_ERROR,
    36: ExitCodes.DRIVER_INPUT_ERROR,
    37: ExitCodes.DRIVER_UNSUPPORTED,
}


class FDErrorInfo:
    def __init__(self, exit_code: ExitCodes, error_message: str):
        self.exit_code = exit_code
        self.error_message = error_message


# NOTE: It is probably sufficient to just have a list instead of this here
#       if there is nothing else that would be interesting about this
class Plan:
    def __init__(self, actionsequence):
        pass


# TODO: In addition to the error parsing here, we should also include parsing for the planning statistics.
#       i.e. how long did the planning take, how many steps etc. if this is a sensible comparison
def parse_planner_output(output: str, returncode: int) -> str | FDErrorInfo:
    fd_code = exit_codes[returncode]
    if (
        fd_code == ExitCodes.SUCCESS
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME
    ):
        # TODO: This will need to be properly parsed once we have plans
        return output
    elif (
        fd_code == ExitCodes.TRANSLATE_CRITICAL_ERROR
        or fd_code == ExitCodes.TRANSLATE_INPUT_ERROR
    ):
        for line in output.strip().split("\n"):
            # TODO: If these are actually relevant after VAL could parse the files, the syntax hints should be included
            # TODO: Figure out whether we want to account for metrics as well
            if (
                line.startswith("Expected a")
                or line.startswith("Reason:")
                or line.startswith("Expecting")
            ):
                return FDErrorInfo(
                    fd_code,
                    f"Error in PDDL translation. Output from fast downward planner:\n{line}",
                )
        return FDErrorInfo(fd_code, "Error in PDDL translation")
    elif (
        fd_code == ExitCodes.SEARCH_UNSOLVABLE
        or fd_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
    ):
        return FDErrorInfo(
            fd_code,
            "Could not find a suitable plan",
        )
    else:
        # TODO: For now we can just pass this on to the model, but it might be an overall good idea to try and rerun
        #       the solver if we run into one of these issues.
        return FDErrorInfo(
            fd_code,
            "Fast Downward encountered an error while trying to generate a plan.",
        )


# TODO: The str output here currently assumes that we have generated a plan
#       → Once the pipeline is capable of producing valid PDDL, there should be an object representing this
#       This function is different from the VAL related functions, as we are also interested in the output
#       if no issues have been found
def generate_plan(domain_file: str, problem_file: str) -> str | FDErrorInfo:
    process = run(
        [
            "python",
            "../fast-downward-24.06.1/fast-downward.py",
            "--alias",
            "seq-sat-lama-2011",
            domain_file,
            problem_file,
        ],
        stderr=PIPE,
        stdout=PIPE,
        text=True,
    )
    return parse_planner_output(process.stdout, process.returncode)
