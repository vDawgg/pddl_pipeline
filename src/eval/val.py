import logging
import re
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, run

logger = logging.getLogger(__name__)


@dataclass
class ValErrorWarningBase:
    error_message: str
    pddl_line: str
    pddl_line_num: int

    def to_str(self) -> str:
        return f"Line {self.pddl_line_num}: {self.pddl_line}\nError Message: {self.error_message}\n"


@dataclass
class ValError(ValErrorWarningBase):
    pass


@dataclass
class ValWarning(ValErrorWarningBase):
    pass


@dataclass
class VALErrorInfo:
    num_errors: int
    num_warnings: int
    errors: list[ValError]
    warnings: list[ValWarning]

    def get_lines_with_errors(self) -> str:
        return "\n".join([e.to_str() for e in self.errors])


def make_error_info(parser_output: str, lines: list[str], problem: bool = False):
    num_errors = 0
    num_warnings = 0
    errors = []
    warnings = []
    for line in parser_output.strip().split("\n"):
        if line.startswith("Errors:"):
            match = re.match(r"Errors:\s*(\d+),\s*warnings:\s*(\d+)", line)
            if match:
                num_errors = int(match.group(1))
                num_warnings = int(match.group(2))

        elif ": Error:" in line or ": Warning:" in line:
            if (
                "domain" in line
                and problem is True
                and "error in problem file" not in line
            ):
                continue
            match = re.search(r"line:\s*(\d+):\s*(Error|Warning):\s*(.+)", line)
            if match:
                line_num = int(match.group(1)) - 1
                error_type = match.group(2)
                if error_type == "Warning":
                    message = match.group(3)
                    warnings.append(
                        ValError(
                            error_message=message.strip(),
                            pddl_line=lines[line_num].strip(),
                            pddl_line_num=line_num,
                        )
                    )
                    continue
                message = match.group(3)
                errors.append(
                    ValError(
                        error_message=message.strip(),
                        pddl_line=lines[line_num].strip(),
                        pddl_line_num=line_num,
                    )
                )
    if len(errors) == 0 and num_errors > 0:
        logger.debug("# Issue in VAL error info")
        logger.debug(parser_output)
    return VALErrorInfo(num_errors, num_warnings, errors, warnings)


def get_syntax_mistakes_domain(domain_file: Path) -> VALErrorInfo:
    process = run(
        ["Parser", domain_file],
        stdout=PIPE,
        text=True,
    )
    with open(domain_file) as f:
        error_info = make_error_info(process.stdout, f.read().split("\n"))
    return error_info


def get_syntax_mistakes_problem(domain_file: Path, problem_file: Path) -> VALErrorInfo:
    process = run(
        ["Parser", domain_file, problem_file],
        stdout=PIPE,
        text=True,
    )
    with open(problem_file) as f:
        error_info = make_error_info(process.stdout, f.read().split("\n"), True)
    return error_info


def is_domain_valid(domain_file: Path) -> bool:
    err_info = get_syntax_mistakes_domain(domain_file)
    if err_info.num_errors > 0:
        logger.debug("Failed to create a syntactically valid domain")
        return False
    logger.debug("Generated syntactically valid domain")
    return True


def is_problem_valid(
    domain_file: Path,
    problem_file: Path,
) -> bool:
    err_info = get_syntax_mistakes_problem(domain_file, problem_file)
    if err_info.num_errors > 0:
        logger.debug("Failed to create a syntactically valid problem")
        return False
    logger.debug("Generated syntactically valid problem")
    return True
