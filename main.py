from src.eval.fast_downward import generate_plan
from src.eval.val import get_syntax_mistakes_problem, get_syntax_mistakes_domain
from src.pipeline.pipeline import baseline
from src.utils.io import write_temp_pddl_file


if __name__ == "__main__":
    domain, problem = baseline()
    print("# Domain\n\n")
    print(domain)
    print("\n\n# Problem\n\n")
    print(problem)
    domain_file = write_temp_pddl_file(domain)
    problem_file = write_temp_pddl_file(problem)
    print("\n\n# Syntax checks domain\n\n")
    print(get_syntax_mistakes_domain(domain_file))
    print("\n\n# Syntax check problem\n\n")
    print(get_syntax_mistakes_problem(domain_file, problem_file))
    print("\n\n# Generated plan\n\n")
    print(generate_plan(domain_file, problem_file))
