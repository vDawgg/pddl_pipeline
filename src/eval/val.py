from subprocess import PIPE, run


def get_syntax_mistakes_domain(domain_file: str) -> str:
    process = run(
        ["./VAL/Parser", domain_file],
        check=True,
        stdout=PIPE,
        text=True,
    )
    # TODO: Later on the output should be parsed to include the number of syntax errors as well
    #       instead of only the syntax error from the single file.
    return process.stdout


def get_syntax_mistakes_problem(domain_file: str, problem_file: str) -> str:
    process = run(
        ["./VAL/Parser", domain_file, problem_file],
        check=True,
        stdout=PIPE,
        text=True,
    )
    # TODO: Same as above
    return process.stdout
