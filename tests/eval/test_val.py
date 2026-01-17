from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from tests.constants import eval_resource_dir


class TestVAL:
    def test_domain_no_err(self):
        error_info = get_syntax_mistakes_domain(
            eval_resource_dir / "test_domain_no_err" / "domain.pddl",
        )
        assert error_info.num_errors == 0

    def test_domain_err(self):
        error_info = get_syntax_mistakes_domain(
            eval_resource_dir / "test_domain_err" / "domain.pddl",
        )
        assert error_info.num_errors == 3
        assert len(error_info.errors) == 3
        assert error_info.errors[0].pddl_line == ":parameters (?x - block)"

    def test_domain_end_of_file_error_line(self):
        error_info = get_syntax_mistakes_domain(
            eval_resource_dir / "test_domain_end_of_file_error_line" / "domain.pddl",
        )
        assert error_info.num_errors == 1
        assert all(err.pddl_line_num in [26] for err in error_info.errors)

    def test_problem_no_err(self):
        error_info = get_syntax_mistakes_problem(
            eval_resource_dir / "test_problem_no_err" / "domain.pddl",
            eval_resource_dir / "test_problem_no_err" / "problem.pddl",
        )
        assert error_info.num_errors == 0

    def test_problem_err(self):
        error_info = get_syntax_mistakes_problem(
            eval_resource_dir / "test_problem_err" / "domain.pddl",
            eval_resource_dir / "test_problem_err" / "problem.pddl",
        )
        assert error_info.num_errors == 1
        assert len(error_info.errors) == 1
        assert error_info.errors[0].pddl_line == "(:start"

    def test_problem_warning_in_domain(self):
        error_info = get_syntax_mistakes_problem(
            eval_resource_dir / "test_problem_end_of_file_error_line" / "domain.pddl",
            eval_resource_dir / "test_problem_end_of_file_error_line" / "problem.pddl",
        )
        assert error_info.num_errors == 0
