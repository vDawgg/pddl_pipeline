import difflib

from src.base.pipeline import PipelineBase
from src.base.schemas import Domains, PDDLFiles, Problems
from src.eval.fast_downward import (
    ExitCodes,
    FDErrorInfo,
    UnsolvabilityFeedback,
)
from src.inference import Models
from tests.constants import eval_resource_dir


class TestFastDownward:
    def setup_method(self):
        self.pipeline = PipelineBase(
            model=Models.GEMMA_3_12B,
            domain=Domains.RING_AND_PEG,
            problem=Problems.RING_AND_PEG_1,
        )

    def test_translate_error_expected(self):
        test_resources = eval_resource_dir / "test_translate_error_expected"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()

    def test_translate_error_incorrect_domain_start(self):
        test_resources = (
            eval_resource_dir / "test_translate_error_incorrect_domain_start"
        )
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert error_info.file == PDDLFiles.DOMAIN
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()

    def test_duplicate_objects(self):
        test_resources = eval_resource_dir / "test_duplicate_objects"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.TRANSLATE_INPUT_ERROR
        assert error_info.file == PDDLFiles.DOMAIN
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()

    def test_search_unsolved_curated(self):
        test_resources = eval_resource_dir / "test_search_unsolved_curated"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
            UnsolvabilityFeedback.CURATED,
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()

    def test_search_unsolved_full(self):
        test_resources = eval_resource_dir / "test_search_unsolved_full"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
            UnsolvabilityFeedback.FULL,
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        with open(test_resources / "out.txt") as f:
            # Stats and temp files will vary from run to run.
            # Therefor doing best effort string matching.
            assert (
                difflib.SequenceMatcher(None, f.read(), error_info.to_str()).ratio()
                >= 0.4
            )

    def test_search_unsolved_simple(self):
        test_resources = eval_resource_dir / "test_search_unsolved_simple"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()

    def test_abstraction_generation(self):
        test_resources = eval_resource_dir / "test_abstraction_generation"
        error_info = self.pipeline._generate_plan(
            (test_resources / "domain.pddl").as_posix(),
            (test_resources / "problem.pddl").as_posix(),
            UnsolvabilityFeedback.ABSTRACTION,
        )
        assert type(error_info) is FDErrorInfo
        assert error_info.exit_code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
        with open(test_resources / "out.txt") as f:
            assert error_info.to_str() == f.read()
