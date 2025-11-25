from subprocess import PIPE, run


def generate_plan(domain_file: str, problem_file: str) -> str:
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
    return process.stdout
