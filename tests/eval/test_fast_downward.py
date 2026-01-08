from src.inference import Models
from src.pipeline import Pipelines
from src.utils.domains import Domains
from src.eval.fast_downward import (
    generate_plan,
    ExitCodes,
    FDErrorInfo,
)
from tests.constants import eval_resource_dir


class TestFastDownward:
    def test_translate_error_expected(self):
        pddl_dir = eval_resource_dir / "test_translate_error_expected"
        error_info = generate_plan(
            str(pddl_dir / "domain.pddl"),
            str(pddl_dir / "problem.pddl"),
            Models.QWEN_3_VL_8B,
            Pipelines.VAL_AND_PLANNER_FEEDBACK,
            Domains.RING_AND_PEG,
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert (
            error_info.error_message
            == "Error in PDDL translation. Output from fast downward planner:\nExpected a non-empty block starting with any of the following words: :requirements, :types, :constants, :predicates, :functions, :derived, :action\nGot: [':actions', ['pick-up', ':parameters', ['?x', '-', 'block'], ':precondition', ['and', ['on-table', '?x'], ['clear', '?x'], ['free-arm']], ':effect', ['and', ['not', ['on-table', '?x']], ['holding', '?x'], ['not', ['clear', '?x']], ['not', ['free-arm']], ['gripped', '?x']]], ['put-down', ':parameters', ['?x', '-', 'block'], ':precondition', ['and', ['holding', '?x'], ['clear', '?y'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['holding', '?x']], ['on-table', '?x'], ['clear', '?y'], ['not', ['gripped', '?x']], ['free-arm']]], ['stack', ':parameters', ['?x', '-', 'block', '?y', '-', 'block'], ':precondition', ['and', ['holding', '?x'], ['on', '?y', '?z'], ['clear', '?y'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['holding', '?x']], ['on', '?x', '?y'], ['clear', '?z'], ['not', ['gripped', '?x']], ['free-arm'], ['attached', '?x']]], ['unstack', ':parameters', ['?x', '-', 'block', '?y', '-', 'block'], ':precondition', ['and', ['on', '?x', '?y'], ['clear', '?x'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['on', '?x', '?y']], ['on-table', '?x'], ['clear', '?y'], ['not', ['attached', '?x']], ['gripped', '?x'], ['free-arm']]]]"
        )

    def test_search_unsolved(self):
        pddl_dir = eval_resource_dir / "test_search_unsolved"
        error_info = generate_plan(
            str(pddl_dir / "domain.pddl"),
            str(pddl_dir / "problem.pddl"),
            Models.QWEN_3_VL_8B,
            Pipelines.VAL_AND_PLANNER_FEEDBACK,
            Domains.RING_AND_PEG,
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        assert error_info.error_message == "Could not find a suitable plan"
