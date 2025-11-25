from src.eval.val import get_syntax_mistakes_problem, get_syntax_mistakes_domain
from src.pipeline.pipeline import baseline
from src.utils.io import write_temp_pddl_file


if __name__ == "__main__":
    domain, problem = baseline()
    print("## Domain\n\n")
    print(domain)
    print("# Problem\n\n")
    print(problem)
    domain_file = write_temp_pddl_file(domain)
    problem_file = write_temp_pddl_file(problem)
    print("# Syntax checks domain\n\n")
    print(get_syntax_mistakes_domain(domain_file))
    print("# Syntax check problem\n\n")
    print(get_syntax_mistakes_problem(domain_file, problem_file))
