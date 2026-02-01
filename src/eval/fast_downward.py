"""
The parsing code is partially based on the fastdownward parsers from the downward lab project
https://github.com/aibasel/lab/tree/main/downward/parsers
"""

import logging
import re
from enum import StrEnum, auto
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

from src.base.schema import PDDLFiles, PipelineError

logger = logging.getLogger(__name__)


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


def is_translate_error(code: ExitCodes):
    return (
        code == ExitCodes.TRANSLATE_CRITICAL_ERROR
        or code == ExitCodes.TRANSLATE_INPUT_ERROR
    )


def is_unsolvable(code: ExitCodes):
    return (
        code == ExitCodes.SEARCH_UNSOLVABLE
        or code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
    )


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


def detect_file_type(output: str) -> PDDLFiles | None:
    lines = output.split("\n")
    last_file_type = None
    for line in lines:
        stripped = line.strip()
        if (
            stripped == "Parsing domain"
            or stripped.startswith("Parsing domain")
            or "Could not parse domain file" in stripped
        ):
            last_file_type = PDDLFiles.DOMAIN
        elif stripped == "Parsing task" or stripped.startswith("Parsing task"):
            last_file_type = PDDLFiles.PROBLEM
    return last_file_type


class FDErrorInfo:
    def __init__(
        self, exit_code: ExitCodes, error_message: str, file: PDDLFiles | None = None
    ):
        self.exit_code = exit_code
        self.error_message = error_message
        self.file = file

    def to_str(self):
        return f"# Error message: {self.error_message}\n# Affected file: {self.file}\n"

    def to_pipeline_error(self) -> PipelineError:
        if is_translate_error(self.exit_code):
            return PipelineError.PLAN_FAILURE_TRANSLATE
        elif is_unsolvable(self.exit_code):
            return PipelineError.PLAN_FAILURE_UNSOLVABLE
        return PipelineError.PLAN_FAILURE


class TranslateParser:
    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.file_read_error = re.compile(
            r"Error: Could not read file: (.+?)\nReason: (.+)", re.DOTALL
        )
        self.file_parse_error = re.compile(
            r"Error: Could not parse (domain|task) file: (.+?)\nReason: (.+)", re.DOTALL
        )
        self.domain_mismatch = re.compile(
            r"The domain name specified by the task \(([^)]+)\) does not match "
            r"the name specified by the domain file \(([^)]+)\)"
        )
        self.out_of_memory = re.compile(r"Translator ran out of memory")
        self.out_of_time = re.compile(r"Translator hit the time limit")
        self.derived_in_init = re.compile(
            r"error: derived predicate '([^']+)' appears in :init fact '([^']+)'"
        )
        self.derived_in_effect = re.compile(
            r"error: derived predicate '([^']+)' appears in effect of action '([^']+)'"
        )
        self.axiom_error = re.compile(r"axioms are not stratifiable")
        self.object_fluent = re.compile(
            r"Error: object fluents not supported\n\(function (\w+) has type (\w+)\)"
        )
        self.python_version = re.compile(
            r"Error: Translator only supports Python >= (\d+\.\d+)"
        )

    def parse_translate_error(self, output: str, fd_code: ExitCodes) -> FDErrorInfo:
        if fd_code == ExitCodes.TRANSLATE_OUT_OF_MEMORY:
            return FDErrorInfo(fd_code, "Translator ran out of memory")
        elif fd_code == ExitCodes.TRANSLATE_OUT_OF_TIME:
            return FDErrorInfo(fd_code, "Translator hit the time limit")
        output = output.strip()
        if match := self.file_read_error.search(output):
            reason = match.group(2).strip()
            return FDErrorInfo(
                fd_code,
                f"Could not read file: {reason}",
                detect_file_type(output),
            )
        elif match := self.file_parse_error.search(output):
            file_type = match.group(1)
            reason = match.group(3).strip()
            return FDErrorInfo(
                fd_code,
                f"Could not parse {file_type} file: {reason}",
                detect_file_type(output),
            )
        elif match := self.domain_mismatch.search(output):
            task_domain = match.group(1)
            file_domain = match.group(2)
            return FDErrorInfo(
                fd_code,
                f"Domain name mismatch: task expects '{task_domain}' but domain file defines '{file_domain}'",
                detect_file_type(output),
            )
        elif match := self.derived_in_init.search(output):
            return FDErrorInfo(
                fd_code,
                f"Derived predicate '{match.group(1)}' appears in :init fact '{match.group(2)}'",
                detect_file_type(output),
            )
        elif match := self.derived_in_effect.search(output):
            return FDErrorInfo(
                fd_code,
                f"Derived predicate '{match.group(1)}' appears in effect of action '{match.group(2)}'",
                detect_file_type(output),
            )
        elif self.axiom_error.search(output):
            return FDErrorInfo(
                fd_code,
                "Axioms are not stratifiable",
                detect_file_type(output),
            )
        elif match := self.object_fluent.search(output):
            return FDErrorInfo(
                fd_code,
                f"Object fluents not supported: function '{match.group(1)}' has type '{match.group(2)}'",
                detect_file_type(output),
            )
        return self._parse_semantic_error(output, fd_code)

    def _parse_semantic_error(self, output: str, fd_code: ExitCodes) -> FDErrorInfo:
        lines = output.split("\n")
        error_lines = []
        in_error = False
        skip_patterns = [
            lambda s: s == "Parsing...",
            lambda s: s.startswith("->") or s.startswith("\t->"),
            lambda s: s.startswith("Parsing ") and not in_error,
            lambda s: "[" in s and "wall-clock]" in s,
            lambda s: s.startswith("Translator "),
            lambda s: s.startswith("Generated "),
            lambda s: s.startswith("Computing "),
            lambda s: s.startswith("Building "),
            lambda s: s.startswith("Writing "),
            lambda s: s.startswith("Normalizing "),
            lambda s: s.startswith("Instantiating"),
            lambda s: s.startswith("Preparing "),
            lambda s: s.startswith("Completing "),
            lambda s: s.startswith("Detecting "),
            lambda s: s.startswith("Reordering "),
            lambda s: s.startswith("Processing "),
            lambda s: s.startswith("Simplifying "),
            lambda s: s.startswith("Collecting "),
            lambda s: s.startswith("Choosing "),
            lambda s: s.startswith("Finding "),
            lambda s: s.startswith("Checking "),
            lambda s: s == "Done!",
            lambda s: s.isdigit() or (s.replace(" ", "").replace(".", "").isdigit()),
            lambda s: s.startswith("INFO"),
            lambda s: s.startswith("Driver aborting"),
            lambda s: s.startswith("translate exit code"),
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # TODO: This could probably be implemented a lot cleaner
            should_skip = False
            for pattern in skip_patterns:
                if pattern(stripped):
                    should_skip = True
                    break
            if should_skip:
                continue
            in_error = True
            error_lines.append(stripped)
        assert len(error_lines) > 0
        return FDErrorInfo(
            fd_code,
            "\n".join(error_lines),
            detect_file_type(output),
        )


translate_parser = TranslateParser()


def translate_pddl(domain_file: Path, problem_file: Path) -> FDErrorInfo | None:
    """
    Executes the translation part of FD only.
    This is intended as a separate feedback step before plan-generation to find
    issues in the PDDL that could not be found using VAL.
    After this succceeds without issues, teh only issue that can still occur is
    the domain being unsolvable.
    """
    sas_file = NamedTemporaryFile(delete=False)
    process = run(
        [
            "python",
            "../fast-downward-24.06.1/src/translate/translate.py",
            "--sas-file",
            sas_file.name,
            domain_file,
            problem_file,
        ],
        capture_output=True,
        text=True,
    )
    fd_code = exit_codes[process.returncode]
    if fd_code != ExitCodes.SUCCESS:
        return translate_parser.parse_translate_error(process.stdout, fd_code)
    return None


def parse_error(
    fd_code: ExitCodes, domain_file: Path, problem_file: Path
) -> FDErrorInfo:
    if is_translate_error(fd_code):
        # Run translator only on the same files to get better output
        translate_output = translate_pddl(domain_file, problem_file)
        assert translate_output is not None
        return translate_output
    elif is_unsolvable(fd_code):
        return FDErrorInfo(
            fd_code,
            "Could not find a suitable plan",
        )
    return FDErrorInfo(
        fd_code,
        "Fast Downward encountered an error while trying to generate a plan.",
    )
