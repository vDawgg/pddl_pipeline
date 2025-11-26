from subprocess import PIPE, run
import re
from typing import Tuple


class ErrorInfo:
    def __init__(self, num_errors: int, num_warnings: int, errors: list[str]):
        self.num_errors = num_errors
        self.num_warnings = num_warnings
        self.errors = errors


def make_error_info(parser_output: str):
    num_errors = 0
    num_warnings = 0
    errors = []
    for line in parser_output.strip().split("\n"):
        if line.startswith("Errors:"):
            match = re.match(r"Errors:\s*(\d+),\s*warnings:\s*(\d+)", line)
            if match:
                num_errors = int(match.group(1))
                num_warnings = int(match.group(2))

        elif ": Error:" in line or ": Warning:" in line:
            match = re.search(r"line:\s*(\d+):\s*(Error|Warning):\s*(.+)", line)
            if match:
                # TODO: The line error information could also be enriched with the actual
                #       line it refers to, if the model still needs more context
                line_num = match.group(1)
                error_type = match.group(2)
                message = match.group(3)
                errors.append(f"line: {line_num}: {error_type}: {message}")
    return ErrorInfo(num_errors, num_warnings, errors)


def get_syntax_mistakes_domain(domain_file: str) -> ErrorInfo:
    process = run(
        ["Parser", domain_file],
        stdout=PIPE,
        text=True,
    )
    return make_error_info(process.stdout)


def get_syntax_mistakes_problem(domain_file: str, problem_file: str) -> ErrorInfo:
    process = run(
        ["Parser", domain_file, problem_file],
        stdout=PIPE,
        text=True,
    )
    # TODO: Double check that using the same parsing approach here does not loose needed information
    # NOTE: Unclear how the error information here should be related to the domain
    #       However, this might not really matter, if we assume that we can only continue with a syntactically correct domain
    return make_error_info(process.stdout)
