from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from tests.constants import eval_resource_dir


class TestVAL:
    def test_domain_no_err(self):
        error_info = get_syntax_mistakes_domain(
            str(eval_resource_dir / "test_domain_no_err" / "domain.pddl"),
        )
        assert error_info.num_errors == 0

    def test_domain_err(self):
        error_info = get_syntax_mistakes_domain(
            str(eval_resource_dir / "test_domain_err" / "domain.pddl"),
        )
        assert error_info.num_errors == 3
        assert len(error_info.errors) == 3
        assert len(set(error_info.errors)) == len(error_info.errors)

    def test_problem_no_err(self):
        error_info = get_syntax_mistakes_problem(
            str(eval_resource_dir / "test_problem_no_err" / "domain.pddl"),
            str(eval_resource_dir / "test_problem_no_err" / "problem.pddl"),
        )
        assert error_info.num_errors == 0

    def test_problem_err(self):
        error_info = get_syntax_mistakes_problem(
            str(eval_resource_dir / "test_problem_err" / "domain.pddl"),
            str(eval_resource_dir / "test_problem_err" / "problem.pddl"),
        )
        assert error_info.num_errors == 1
        assert len(error_info.errors) == 1
