import re
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, run


@dataclass
class ValError:
    error_message: str
    pddl_line: str

    def get_line_with_error(self) -> str:
        return f"Line: {self.pddl_line}\nError Message: {self.error_message}\n"


@dataclass
class VALErrorInfo:
    num_errors: int
    num_warnings: int
    errors: list[ValError]

    def get_lines_with_errors(self) -> str:
        return "\n".join([e.get_line_with_error() for e in self.errors])


def make_error_info(parser_output: str, lines: list[str]):
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
                line_num = int(match.group(1)) - 1
                error_type = match.group(2)
                if error_type == "Warning":
                    continue
                message = match.group(3)
                errors.append(
                    ValError(
                        error_message=message.strip(),
                        pddl_line=lines[line_num].strip(),
                    )
                )
    return VALErrorInfo(num_errors, num_warnings, errors)


def get_syntax_mistakes_domain(domain_file: Path) -> VALErrorInfo:
    process = run(
        ["Parser", domain_file],
        stdout=PIPE,
        text=True,
    )
    with open(domain_file) as f:
        error_info = make_error_info(process.stdout, f.readlines())
    return error_info


def get_syntax_mistakes_problem(domain_file: Path, problem_file: Path) -> VALErrorInfo:
    process = run(
        ["Parser", domain_file, problem_file],
        stdout=PIPE,
        text=True,
    )
    with open(problem_file) as f:
        error_info = make_error_info(process.stdout, f.readlines())
    return error_info
