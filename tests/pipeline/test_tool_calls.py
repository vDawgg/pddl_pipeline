import shutil
import tempfile
from pathlib import Path

from src.inference import Models
from src.pipeline.tool_calls import ToolCallPipeline
from src.utils.domains import Domains
from tests.constants import pipeline_resource_dir


class TestToolCallPipeline:
    def setup_method(self):
        self.pipeline = ToolCallPipeline(
            model=Models.GEMMA_3_12B, domain=Domains.RING_AND_PEG
        )

    def test_read_pddl_file_full_content(self):
        file = pipeline_resource_dir / "sample_domain.pddl"
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(str(file))
        with open(file_with_line_numbers) as f:
            assert content == f.read()
            assert self.pipeline.read_pddl_file_calls == 1

    def test_read_pddl_file_with_line_range(self):
        file = pipeline_resource_dir / "sample_domain.pddl"
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(str(file), line_range=(0, 2))
        with open(file_with_line_numbers) as f:
            assert content == "".join(f.readlines()[0:3])
            assert self.pipeline.read_pddl_file_calls == 1

    def test_read_pddl_file_single_line(self):
        file = pipeline_resource_dir / "sample_domain.pddl"
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(str(file), line_range=(0, 0))
        with open(file_with_line_numbers) as f:
            assert content == f.readlines()[0]
            assert self.pipeline.read_pddl_file_calls == 1

    def test_read_pddl_file_increments_call_counter(self):
        file_path = pipeline_resource_dir / "sample_domain.pddl"
        self.pipeline.read_pddl_file(str(file_path))
        assert self.pipeline.read_pddl_file_calls == 1
        self.pipeline.read_pddl_file(str(file_path))
        assert self.pipeline.read_pddl_file_calls == 2
        self.pipeline.read_pddl_file(str(file_path), line_range=(0, 5))
        assert self.pipeline.read_pddl_file_calls == 3

    # edit_lines tests

    def test_edit_lines_single_line_replacement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            self.pipeline.edit_lines(
                str(tmp_file),
                line_range=(4, 4),
                new="  (:types block robot)\n  (:action dummy)",
            )
            with open(tmp_file) as f:
                edited_content = f.read()
            with open(
                pipeline_resource_dir / "test_edited_lines_single_line_replacement.pddl"
            ) as f:
                assert edited_content == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_shrink_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            new_predicates = (
                """(:predicates\n(on ?b1 ?b2)\n(clear ?b)\n(arm-empty)\n)"""
            )
            self.pipeline.edit_lines(
                str(tmp_file), line_range=(6, 11), new=new_predicates
            )
            with open(tmp_file) as f:
                edited_content = f.read()
            with open(
                pipeline_resource_dir / "test_edit_lines_shrink_content.pddl"
            ) as f:
                assert edited_content == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_expand_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            expanded_content = """(:types\nblock\nrobot\ntable\n)"""
            self.pipeline.edit_lines(
                str(tmp_file), line_range=(4, 4), new=expanded_content
            )
            with open(tmp_file) as f:
                content = f.read()
            with open(
                pipeline_resource_dir / "test_edit_lines_expand_content.pddl"
            ) as f:
                assert content == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_first_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            self.pipeline.edit_lines(
                str(tmp_file),
                line_range=(0, 0),
                new="(define (domain new-blocks)",
            )
            with open(tmp_file) as f:
                content = f.read()
            with open(pipeline_resource_dir / "test_edit_lines_first_line.pddl") as f:
                assert content == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_introduce_error_domain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            edit_out = self.pipeline.edit_lines(
                str(tmp_file),
                line_range=(6, 6),
                new="(:actions (pick-up",
            )
            with open(
                pipeline_resource_dir / "test_edit_lines_introduce_error_domain.txt"
            ) as f:
                assert edit_out == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_introduce_error_problem(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_problem.pddl"
            tmp_file = Path(tmp_dir) / "test_problem.pddl"
            shutil.copy(src_file, tmp_file)
            edit_out = self.pipeline.edit_lines(
                str(tmp_file),
                line_range=(8, 8),
                new="(:start",
            )
            with open(
                pipeline_resource_dir / "test_edit_lines_introduce_error_problem.txt"
            ) as f:
                assert edit_out == f.read()
            assert self.pipeline.edit_lines_calls == 1

    def test_edit_lines_increments_call_counter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            tmp_file = Path(tmp_dir) / "test_domain.pddl"
            shutil.copy(src_file, tmp_file)
            self.pipeline.edit_lines(str(tmp_file), line_range=(0, 0), new="")
            assert self.pipeline.edit_lines_calls == 1
            self.pipeline.edit_lines(str(tmp_file), line_range=(1, 1), new="")
            assert self.pipeline.edit_lines_calls == 2
            self.pipeline.edit_lines(str(tmp_file), line_range=(2, 2), new="")
            assert self.pipeline.edit_lines_calls == 3
