from src.base.schema import PDDLFiles
from src.eval.fast_downward import (
    ExitCodes,
    FDErrorInfo,
)
from src.inference import Models
from src.pipeline import Baseline
from src.utils.domains import Domains
from tests.constants import eval_resource_dir


class TestFastDownward:
    def setup_method(self):
        self.pipeline = Baseline(
            model=Models.GEMMA_3_12B,
            domain=Domains.RING_AND_PEG,
        )

    def test_translate_error_expected(self):
        pddl_dir = eval_resource_dir / "test_translate_error_expected"
        error_info = self.pipeline._generate_plan(
            pddl_dir / "domain.pddl",
            pddl_dir / "problem.pddl",
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        print(error_info.error_message)
        assert (
            error_info.error_message
            == "Expected a non-empty block starting with any of the following words: :requirements, :types, :constants, :predicates, :functions, :derived, :action\nGot: [':actions', ['pick-up', ':parameters', ['?x', '-', 'block'], ':precondition', ['and', ['on-table', '?x'], ['clear', '?x'], ['free-arm']], ':effect', ['and', ['not', ['on-table', '?x']], ['holding', '?x'], ['not', ['clear', '?x']], ['not', ['free-arm']], ['gripped', '?x']]], ['put-down', ':parameters', ['?x', '-', 'block'], ':precondition', ['and', ['holding', '?x'], ['clear', '?y'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['holding', '?x']], ['on-table', '?x'], ['clear', '?y'], ['not', ['gripped', '?x']], ['free-arm']]], ['stack', ':parameters', ['?x', '-', 'block', '?y', '-', 'block'], ':precondition', ['and', ['holding', '?x'], ['on', '?y', '?z'], ['clear', '?y'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['holding', '?x']], ['on', '?x', '?y'], ['clear', '?z'], ['not', ['gripped', '?x']], ['free-arm'], ['attached', '?x']]], ['unstack', ':parameters', ['?x', '-', 'block', '?y', '-', 'block'], ':precondition', ['and', ['on', '?x', '?y'], ['clear', '?x'], ['not', ['attached', '?x']]], ':effect', ['and', ['not', ['on', '?x', '?y']], ['on-table', '?x'], ['clear', '?y'], ['not', ['attached', '?x']], ['gripped', '?x'], ['free-arm']]]]"
        )

    def test_translate_error_incorrect_domain_start(self):
        pddl_dir = eval_resource_dir / "test_translate_error_incorrect_domain_start"
        error_info = self.pipeline._generate_plan(
            pddl_dir / "domain.pddl",
            pddl_dir / "problem.pddl",
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert error_info.file == PDDLFiles.DOMAIN
        assert (
            error_info.error_message
            == "Domain definition expected to start with '(define '. Got '(:domain'"
        )

    def test_duplicate_objects(self):
        pddl_dir = eval_resource_dir / "test_duplicate_objects"
        error_info = self.pipeline._generate_plan(
            pddl_dir / "domain.pddl",
            pddl_dir / "problem.pddl",
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert error_info.file == PDDLFiles.DOMAIN
        assert (
            error_info.error_message
            == "Domain contains :constants referenced again in problem. Prefer object definition in problem only.error: duplicate object 'peg1'\n"
            + "error: duplicate object 'peg2'\n"
            + "error: duplicate object 'peg3'\n"
            + "error: duplicate object 'peg4'\n"
            + "error: duplicate object 'peg5'\n"
            + "error: duplicate object 'ring-red'\n"
            + "error: duplicate object 'ring-blue'\n"
            + "error: duplicate object 'ring-green'\n"
            + "error: duplicate object 'ring-yellow'\n"
            + "error: duplicate object 'ring-purple'"
        )

    def test_search_unsolved(self):
        pddl_dir = eval_resource_dir / "test_search_unsolved"
        error_info = self.pipeline._generate_plan(
            pddl_dir / "domain.pddl",
            pddl_dir / "problem.pddl",
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        assert error_info.error_message == "Could not find a suitable plan"
