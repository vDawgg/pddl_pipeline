from src.eval.fast_downward import (
    generate_plan,
    parse_plan,
    Action,
    Plan,
    ExitCodes,
    FDErrorInfo,
)
from tests.constants import eval_resource_dir


class TestFastDownward:
    def test_parse_plan(self):
        plan_true = Plan(
            [
                Action("unstack", ["e", "g"]),
                Action("put-down", ["e"]),
                Action("unstack", ["g", "b"]),
            ]
        )
        plan_parsed = parse_plan(
            str(eval_resource_dir / "test_parse_plan" / "plan.pddl")
        )
        assert plan_true == plan_parsed

    def test_parse_plan_multiple_valid(self):
        plan_true = Plan(
            [
                Action("unstack", ["e", "g"]),
                Action("put-down", ["e"]),
            ]
        )
        plan_parsed = parse_plan(
            str(eval_resource_dir / "test_parse_plan_multiple_valid" / "plan.pddl")
        )
        assert plan_true == plan_parsed

    def test_translate_error_expected(self):
        pddl_dir = eval_resource_dir / "test_translate_error_expected"
        error_info = generate_plan(
            str(pddl_dir / "domain.pddl"),
            str(pddl_dir / "problem.pddl"),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert (
            error_info.error_message
            == "Error in PDDL translation. Output from fast downward planner:\nExpected a non-empty block starting with any of the following words: :requirements, :types, :constants, :predicates, :functions, :derived, :action"
        )

    def test_search_unsolved(self):
        pddl_dir = eval_resource_dir / "test_search_unsolved"
        error_info = generate_plan(
            str(pddl_dir / "domain.pddl"),
            str(pddl_dir / "problem.pddl"),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        assert error_info.error_message == "Could not find a suitable plan"
