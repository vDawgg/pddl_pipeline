import shutil
import tempfile
from pathlib import Path

from src.base.schemas import Domains, PDDLFiles, Problems
from src.inference import Models
from src.pipeline.tool_call import ToolCallPipeline
from tests.constants import pipeline_resource_dir


class TestToolCallPipeline:
    def setup_method(self):
        self.pipeline = ToolCallPipeline(
            model=Models.GEMMA_3_12B,
            domain=Domains.RING_AND_PEG,
            problem=Problems.RING_AND_PEG_1,
        )

    def test_read_pddl_file_full_content(self):
        self.pipeline.vars.domain_file = (
            pipeline_resource_dir / "sample_domain.pddl"
        ).as_posix()
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(PDDLFiles.DOMAIN)
        with open(file_with_line_numbers) as f:
            assert content == f.read()
            assert self.pipeline.vars.read_pddl_file_calls == 1

    def test_read_pddl_file_with_line_range(self):
        self.pipeline.vars.domain_file = (
            pipeline_resource_dir / "sample_domain.pddl"
        ).as_posix()
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(PDDLFiles.DOMAIN, line_range=(0, 2))
        with open(file_with_line_numbers) as f:
            assert content == "".join(f.readlines()[0:3])
            assert self.pipeline.vars.read_pddl_file_calls == 1

    def test_read_pddl_file_single_line(self):
        self.pipeline.vars.domain_file = (
            pipeline_resource_dir / "sample_domain.pddl"
        ).as_posix()
        file_with_line_numbers = (
            pipeline_resource_dir / "sample_domain_line_numbers.pddl"
        )
        content = self.pipeline.read_pddl_file(PDDLFiles.DOMAIN, line_range=(0, 0))
        with open(file_with_line_numbers) as f:
            assert content == f.readlines()[0]
            assert self.pipeline.vars.read_pddl_file_calls == 1

    def test_read_pddl_file_increments_call_counter(self):
        self.pipeline.vars.domain_file = (
            pipeline_resource_dir / "sample_domain.pddl"
        ).as_posix()
        self.pipeline.read_pddl_file(PDDLFiles.DOMAIN)
        assert self.pipeline.vars.read_pddl_file_calls == 1
        self.pipeline.read_pddl_file(PDDLFiles.DOMAIN)
        assert self.pipeline.vars.read_pddl_file_calls == 2
        self.pipeline.read_pddl_file(PDDLFiles.DOMAIN, line_range=(0, 5))
        assert self.pipeline.vars.read_pddl_file_calls == 3

    # edit_lines tests

    def test_edit_lines_single_line_replacement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            self.pipeline.edit_lines(
                PDDLFiles.DOMAIN,
                line_range=(4, 4),
                new="  (:types block robot)\n  (:action dummy)",
            )
            with open(self.pipeline.vars.domain_file) as f:
                edited_content = f.read()
            with open(
                pipeline_resource_dir / "test_edited_lines_single_line_replacement.pddl"
            ) as f:
                assert edited_content == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_shrink_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            new_predicates = (
                """(:predicates\n(on ?b1 ?b2)\n(clear ?b)\n(arm-empty)\n)"""
            )
            self.pipeline.edit_lines(
                PDDLFiles.DOMAIN, line_range=(6, 11), new=new_predicates
            )
            with open(self.pipeline.vars.domain_file) as f:
                edited_content = f.read()
            with open(
                pipeline_resource_dir / "test_edit_lines_shrink_content.pddl"
            ) as f:
                assert edited_content == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_expand_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            expanded_content = """(:types\nblock\nrobot\ntable\n)"""
            self.pipeline.edit_lines(
                PDDLFiles.DOMAIN, line_range=(4, 4), new=expanded_content
            )
            with open(self.pipeline.vars.domain_file) as f:
                content = f.read()
            with open(
                pipeline_resource_dir / "test_edit_lines_expand_content.pddl"
            ) as f:
                assert content == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_first_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            self.pipeline.edit_lines(
                PDDLFiles.DOMAIN,
                line_range=(0, 0),
                new="(define (domain new-blocks)",
            )
            with open(self.pipeline.vars.domain_file) as f:
                content = f.read()
            with open(pipeline_resource_dir / "test_edit_lines_first_line.pddl") as f:
                assert content == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_introduce_error_domain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            edit_out = self.pipeline.edit_lines(
                PDDLFiles.DOMAIN,
                line_range=(6, 6),
                new="(:actions (pick-up",
            )
            with open(
                pipeline_resource_dir / "test_edit_lines_introduce_error_domain.txt"
            ) as f:
                assert edit_out == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_introduce_error_problem(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_problem.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_problem.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            edit_out = self.pipeline.edit_lines(
                PDDLFiles.DOMAIN,
                line_range=(8, 8),
                new="(:start",
            )
            with open(
                pipeline_resource_dir / "test_edit_lines_introduce_error_problem.txt"
            ) as f:
                assert edit_out == f.read()
            assert self.pipeline.vars.edit_lines_calls == 1

    def test_edit_lines_increments_call_counter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = pipeline_resource_dir / "sample_domain.pddl"
            self.pipeline.vars.domain_file = (
                Path(tmp_dir) / "test_domain.pddl"
            ).as_posix()
            shutil.copy(src_file, self.pipeline.vars.domain_file)
            self.pipeline.edit_lines(PDDLFiles.DOMAIN, line_range=(0, 0), new="")
            assert self.pipeline.vars.edit_lines_calls == 1
            self.pipeline.edit_lines(PDDLFiles.DOMAIN, line_range=(1, 1), new="")
            assert self.pipeline.vars.edit_lines_calls == 2
            self.pipeline.edit_lines(PDDLFiles.DOMAIN, line_range=(2, 2), new="")
            assert self.pipeline.vars.edit_lines_calls == 3
