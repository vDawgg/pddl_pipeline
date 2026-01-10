"""
The parsing code is partially based on the fastdownward parsers from the downward lab project
https://github.com/aibasel/lab/tree/main/downward/parsers
"""

import shutil
from pathlib import Path
from enum import StrEnum, auto

from subprocess import PIPE, run
from tempfile import NamedTemporaryFile

from src.base.schema import PDDLFiles
from src.constants import plans_dir
from src.utils.timestamp import get_current_timestamp


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


def save_plan(plan_file: Path, name: str) -> Path:
    latest_plan = ""
    for i in range(1, 100):
        candidate_path = Path(f"{plan_file}.{i}")
        if candidate_path.is_file():
            latest_plan = candidate_path
        else:
            break
    plan_name = plans_dir / f"{name}_{get_current_timestamp()}.plan"
    shutil.copyfile(latest_plan, plan_name)
    return plan_name


class FDErrorInfo:
    def __init__(
        self, exit_code: ExitCodes, error_message: str, file: PDDLFiles | None = None
    ):
        self.exit_code = exit_code
        self.error_message = error_message
        self.file = file


def parse_error(fd_code: ExitCodes, output: str) -> FDErrorInfo:
    if (
        fd_code == ExitCodes.TRANSLATE_CRITICAL_ERROR
        or fd_code == ExitCodes.TRANSLATE_INPUT_ERROR
    ):
        current_file = None
        lines = output.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Parsing domain") or line.startswith(
                "Error: Could not parse domain file"
            ):
                current_file = PDDLFiles.DOMAIN
            elif line.startswith("Parsing problem"):
                current_file = PDDLFiles.PROBLEM
            if (
                line.startswith("Expected a")
                or line.startswith("Reason:")
                or line.startswith("Expecting")
            ):
                assert i + 1 < len(lines)
                return FDErrorInfo(
                    fd_code,
                    f"Error in PDDL translation. Output from fast downward planner:\n{line}\n{lines[i + 1]}",
                    current_file,
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


def generate_plan(
    domain_file: Path,
    problem_file: Path,
    name: str,
) -> FDErrorInfo | Path:
    plan_file = NamedTemporaryFile(delete=False)
    process = run(
        [
            "python",
            "../fast-downward-24.06.1/fast-downward.py",
            "--overall-time-limit",
            "1m",
            "--plan-file",
            plan_file.name,
            "--alias",
            "seq-sat-lama-2011",
            domain_file,
            problem_file,
        ],
        stderr=PIPE,
        stdout=PIPE,
        text=True,
    )
    fd_code = exit_codes[process.returncode]
    # TODO: In addition to the error parsing here, we should also include parsing for the planning statistics.
    #       i.e. how long did the planning take, how many steps etc. if this is a sensible comparison
    if (
        fd_code == ExitCodes.SUCCESS
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME
        or fd_code == ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME
    ):
        return save_plan(Path(plan_file.name), name)
    else:
        return parse_error(fd_code, process.stdout)
