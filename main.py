from src.eval.fast_downward import generate_plan
from src.eval.val import get_syntax_mistakes_problem, get_syntax_mistakes_domain
from src.pipeline.pipeline import baseline
from src.utils.io import write_temp_pddl_file


if __name__ == "__main__":
    domain, problem = baseline()
    print("# Domain\n\n")
    print(domain)
    domain_file = write_temp_pddl_file(domain)
    print("\n\n# Syntax checks domain\n\n")
    domain_error_info = get_syntax_mistakes_domain(domain_file)
    print("Num errors:", domain_error_info.num_errors)
    print("Num warnings:", domain_error_info.num_warnings)
    print("Errors: ", domain_error_info.errors)
    print("\n\n# Problem\n\n")
    print(problem)
    problem_file = write_temp_pddl_file(problem)
    print("\n\n# Syntax check problem\n\n")
    problem_error_info = get_syntax_mistakes_problem(domain_file, problem_file)
    print("Num errors:", problem_error_info.num_errors)
    print("Num warnings:", problem_error_info.num_warnings)
    print("Errors: ", problem_error_info.errors)
    print("\n\n# Generated plan\n\n")
    print(generate_plan(domain_file, problem_file))
