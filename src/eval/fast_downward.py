import logging
import re
from enum import Enum, StrEnum, auto
from itertools import product
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

from pddl.core import Action, Domain, Formula, Problem
from pddl.logic.base import And, BinaryOp, Not, QuantifiedCondition, UnaryOp
from pddl.logic.predicates import EqualTo, Predicate

from pddl import parse_domain, parse_problem
from src.base.schema import PDDLFiles, PipelineError

logger = logging.getLogger(__name__)


class ExitCodes(StrEnum):
    # Successfull terminations
    SUCCESS = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_TIME = auto()
    SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME = auto()
    # Unsuccessfull but error free termination
    TRANSLATE_UNSOLVABLE = auto()
    SEARCH_UNSOLVABLE = auto()
    SEARCH_UNSOLVED_INCOMPLETE = auto()
    # Expected failures
    TRANSLATE_OUT_OF_MEMORY = auto()
    TRANSLATE_OUT_OF_TIME = auto()
    SEARCH_OUT_OF_MEMORY = auto()
    SEARCH_OUT_OF_TIME = auto()
    SEARCH_OUT_OF_MEMORY_AND_TIME = auto()
    # Unrecoverable failures
    TRANSLATE_CRITICAL_ERROR = auto()
    TRANSLATE_INPUT_ERROR = auto()
    SEARCH_CRITICAL_ERROR = auto()
    SEARCH_INPUT_ERROR = auto()
    SEARCH_UNSUPPORTED = auto()
    DRIVER_CRITICAL_ERROR = auto()
    DRIVER_INPUT_ERROR = auto()
    DRIVER_UNSUPPORTED = auto()


def is_translate_error(code: ExitCodes):
    return (
        code == ExitCodes.TRANSLATE_CRITICAL_ERROR
        or code == ExitCodes.TRANSLATE_INPUT_ERROR
    )


def is_unsolvable(code: ExitCodes):
    return (
        code == ExitCodes.SEARCH_UNSOLVABLE
        or code == ExitCodes.SEARCH_UNSOLVED_INCOMPLETE
    )


class UnsolvabilityFeedback(Enum):
    SIMPLE = auto()
    CURATED = auto()
    FULL = auto()
    ABSTRACTION = auto()


exit_codes = {
    0: ExitCodes.SUCCESS,
    1: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY,
    2: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME,
    3: ExitCodes.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME,
    10: ExitCodes.TRANSLATE_UNSOLVABLE,
    11: ExitCodes.SEARCH_UNSOLVABLE,
    12: ExitCodes.SEARCH_UNSOLVED_INCOMPLETE,
    20: ExitCodes.TRANSLATE_OUT_OF_MEMORY,
    21: ExitCodes.TRANSLATE_OUT_OF_TIME,
    22: ExitCodes.SEARCH_OUT_OF_MEMORY,
    23: ExitCodes.SEARCH_OUT_OF_TIME,
    24: ExitCodes.SEARCH_OUT_OF_MEMORY_AND_TIME,
    30: ExitCodes.TRANSLATE_CRITICAL_ERROR,
    31: ExitCodes.TRANSLATE_INPUT_ERROR,
    32: ExitCodes.SEARCH_CRITICAL_ERROR,
    33: ExitCodes.SEARCH_INPUT_ERROR,
    34: ExitCodes.SEARCH_UNSUPPORTED,
    35: ExitCodes.DRIVER_CRITICAL_ERROR,
    36: ExitCodes.DRIVER_INPUT_ERROR,
    37: ExitCodes.DRIVER_UNSUPPORTED,
}


def detect_file_type(output: str) -> PDDLFiles | None:
    lines = output.split("\n")
    last_file_type = None
    for line in lines:
        stripped = line.strip()
        logger.debug(stripped)
        if (
            stripped == "Parsing domain"
            or stripped.startswith("Parsing domain")
            or "Could not parse domain file" in stripped
        ):
            last_file_type = PDDLFiles.DOMAIN
        elif stripped == "Parsing task" or stripped.startswith("Parsing task"):
            last_file_type = PDDLFiles.PROBLEM
    return last_file_type


class FDErrorInfo:
    def __init__(
        self, exit_code: ExitCodes, error_message: str, file: PDDLFiles | None = None
    ):
        self.exit_code = exit_code
        self.error_message = error_message
        self.file = file

    def to_str(self):
        base = f"# Error message: {self.error_message}\n"
        if self.file is None:
            return base
        return base + f"# Affected file: {self.file}\n"

    def to_pipeline_error(self) -> PipelineError:
        if is_translate_error(self.exit_code):
            return PipelineError.PLAN_FAILURE_TRANSLATE
        elif is_unsolvable(self.exit_code):
            return PipelineError.PLAN_FAILURE_UNSOLVABLE
        return PipelineError.PLAN_FAILURE


class AbstractionGenerator:
    """
    This class is used to generate abstractions of a given PDDL task.
    Abstraction in this case refers to removing a set of n fluents from the original task.
    This can be useful for discerning what set of predicates might lead to the task not being solvable.

    The basis for the generation of the abstracted tasks is:
    Sreedharan, Sarath, et al. "Why Can't You Do That HAL? Explaining Unsolvability of Planning Tasks." IJCAI. 2019.
    """

    def __init__(self, domain_file: Path, problem_file: Path):
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.parsed_domain = parse_domain(self.domain_file)
        self.parsed_problem = parse_problem(self.problem_file)

    def generate_predicate_combinations(self) -> list[list[set[Predicate]]]:
        """
        Function used for generating all possible predicate combinations that can be
        removed from a pddl domain and problem.
        """
        predicates = self.parsed_domain.predicates
        # Iteratively build set of n predicate combinations used for later removal from domain
        predicate_removals = []
        for i in range(len(predicates)):
            predicate_lists = [predicates for _ in range(i)]
            unique_combinations = set()
            for predicate_combinations in product(*predicate_lists):
                # Remove all entries where fluents repeat
                if len(predicate_combinations) == len(set(predicate_combinations)):
                    unique_combinations.add(frozenset(predicate_combinations))
            predicate_removals.append(
                [sorted(set(combo)) for combo in unique_combinations if len(combo) > 0]
            )
        return predicate_removals

    def build_abstraction(self) -> str | None:
        """
        Build abstracted domain and problems, where a set of predicates has been
        removed and check whether removing these predicates makes the task solvable.
        Returns the removed predicate set if the task was made solvable.
        """
        predicate_subsets = self.generate_predicate_combinations()
        for predicate_group in predicate_subsets:
            for predicates_to_remove in predicate_group:
                removed_predicate_names = {
                    predicate.name for predicate in predicates_to_remove
                }
                domain = self._build_domain(removed_predicate_names)
                problem = self._build_problem(domain, removed_predicate_names)
                domain_file = self._write_temp_pddl(str(domain))
                problem_file = self._write_temp_pddl(str(problem))
                _, fd_code, _ = generate_plan(domain_file, problem_file)
                if fd_code == ExitCodes.SUCCESS:
                    return ", ".join(
                        sorted(str(predicate) for predicate in predicates_to_remove)
                    )
        return None

    def _build_domain(self, removed_predicate_names: set[str]) -> Domain:
        domain = self.parsed_domain
        predicates = {
            predicate
            for predicate in domain.predicates
            if predicate.name not in removed_predicate_names
        }
        actions = set()
        for action in domain.actions:
            precondition = self._filter_formula(
                action.precondition, removed_predicate_names
            )
            effect = self._filter_formula(action.effect, removed_predicate_names)
            # The PDDL lib cant handle None as a precondition.
            # An empty And() is equivalent to an empty precondition here.
            if precondition is None:
                precondition = And()
            if effect is None:
                effect = And()
            actions.add(
                Action(
                    name=action.name,
                    parameters=action.parameters,
                    precondition=precondition,
                    effect=effect,
                )
            )
        return Domain(
            name=domain.name,
            requirements=domain.requirements,
            types=domain.types,  # type: ignore
            constants=domain.constants,
            predicates=predicates,
            functions=domain.functions,
            actions=actions,
        )

    def _build_problem(
        self, domain: Domain, removed_predicate_names: set[str]
    ) -> Problem:
        problem = self.parsed_problem
        init = {
            filtered
            for formula in problem.init
            if (filtered := self._filter_formula(formula, removed_predicate_names))
            is not None
        }
        goal = self._filter_formula(problem.goal, removed_predicate_names)
        if goal is None:
            goal = And()
        return Problem(
            name=problem.name,
            domain=domain,
            objects=problem.objects,
            init=init,
            goal=goal,
            metric=problem.metric,
        )

    def _filter_formula(  # noqa: C901
        self, formula: Formula | None, removed_predicate_names: set[str]
    ) -> Formula | None:
        # Predicates are clustered in formulas when defining e.g. preconditions for
        # actions or the goal in a task in the pddl library. When removing predicates
        # from the general predicate list in domains, we also have to make sure such
        # predicates are no longer mentioned in the domains/problems formulas.
        if formula is None:
            return None
        if isinstance(formula, Predicate):
            return None if formula.name in removed_predicate_names else formula
        if isinstance(formula, EqualTo):
            return formula
        if isinstance(formula, Not):
            argument = self._filter_formula(formula.argument, removed_predicate_names)
            if argument is None:
                return None
            return Not(argument)
        if isinstance(formula, QuantifiedCondition):
            condition = self._filter_formula(formula.condition, removed_predicate_names)
            if condition is None:
                return None
            return type(formula)(condition, formula.variables)
        if isinstance(formula, BinaryOp):
            operands = [
                self._filter_formula(operand, removed_predicate_names)
                for operand in formula.operands
            ]
            filtered_operands = [operand for operand in operands if operand is not None]
            if not filtered_operands:
                return None
            if len(filtered_operands) != len(formula.operands) and not isinstance(
                formula, And
            ):
                return None
            return type(formula)(*filtered_operands)
        if isinstance(formula, UnaryOp):
            argument = self._filter_formula(formula.argument, removed_predicate_names)
            if argument is None:
                return None
            return type(formula)(argument)
        return formula

    def _write_temp_pddl(self, pddl_text: str) -> Path:
        temp_file = NamedTemporaryFile(delete=False, mode="w", suffix=".pddl")
        temp_file.write(pddl_text)
        temp_file.flush()
        temp_file.close()
        return Path(temp_file.name)


class UnsolvabilityParser:
    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.no_relaxed_solution = re.compile(r"No relaxed solution")
        self.trivially_false = re.compile(r"Trivially false goal")
        self.simplified_trivially_false = re.compile(
            r"Simplified to trivially false goal"
        )
        self.empty_goal = re.compile(r"Simplified to empty goal")
        self.relevant_atoms = re.compile(r"(\d+) relevant atoms")
        self.operators_removed = re.compile(r"(\d+) operators removed")
        self.grounded_operators = re.compile(r"Translator operators: (\d+)")

    def parse_unsolvability_hints(self, output: str, fd_code) -> FDErrorInfo:
        err_message = [
            "The task is unsolvable! Use the following error information to fix this.\n"
        ]
        if match := self.no_relaxed_solution.search(output):
            err_message.append(
                "The goal is not reachable by any action sequence. Check the preconditions and effects."
            )
        if match := self.trivially_false.search(output):
            err_message.append(
                "The goal contains a predicate that cannot be grounded. Check the goal predicates."
            )
        if match := self.simplified_trivially_false.search(output):
            err_message.append(
                "The goal contains a predicate that is never satisfied by any available action from the initial state. Check that all goal predicates are affected by at least one action from their init state."
            )
        if match := self.empty_goal.search(output):
            err_message.append(
                "The goal is empty and was either already satisfied in the initial state or was simplified away during planning. Check the goal conditions."
            )
        if match := self.relevant_atoms.search(output):
            num_relevant_atoms = match.group(1)
            # NOTE: If the number here is very strange it might be helpful to add the output generated from
            #       dump_task thought this output might be very large depending on the task
            err_message.append(
                f"The domain contains {num_relevant_atoms} reachable grounded PDDL atoms. Verify that this number matches what you would expect."
            )
        if match := self.operators_removed.search(output):
            num_removed_operators = match.group(1)
            err_message.append(
                f"The planner had to remove {num_removed_operators} actions from the set as they contain either impossible preconditions or no effects. Check the action preconditions and effects."
            )
        if match := self.grounded_operators.search(output):
            num_grounded_operators = match.group(1)
            err_message.append(
                f"The planner generated {num_grounded_operators} actions that could be grounded. Verify that this number matches what you would expect."
            )
        return FDErrorInfo(fd_code, "\n".join(err_message))


class TranslateParser:
    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.file_read_error = re.compile(
            r"Error: Could not read file: (.+?)\nReason: (.+)", re.DOTALL
        )
        self.file_parse_error = re.compile(
            r"Error: Could not parse (domain|task) file: (.+?)\nReason: (.+)", re.DOTALL
        )
        self.domain_mismatch = re.compile(
            r"The domain name specified by the task \(([^)]+)\) does not match "
            r"the name specified by the domain file \(([^)]+)\)"
        )
        self.out_of_memory = re.compile(r"Translator ran out of memory")
        self.out_of_time = re.compile(r"Translator hit the time limit")
        self.derived_in_init = re.compile(
            r"error: derived predicate '([^']+)' appears in :init fact '([^']+)'"
        )
        self.derived_in_effect = re.compile(
            r"error: derived predicate '([^']+)' appears in effect of action '([^']+)'"
        )
        self.axiom_error = re.compile(r"axioms are not stratifiable")
        self.object_fluent = re.compile(
            r"Error: object fluents not supported\n\(function (\w+) has type (\w+)\)"
        )
        self.python_version = re.compile(
            r"Error: Translator only supports Python >= (\d+\.\d+)"
        )
        self.duplicate_object = re.compile(r"error: duplicate .*")

    def parse_translate_error(self, output: str, fd_code: ExitCodes) -> FDErrorInfo:  # noqa: C901
        if fd_code == ExitCodes.TRANSLATE_OUT_OF_MEMORY:
            return FDErrorInfo(fd_code, "Translator ran out of memory")
        elif fd_code == ExitCodes.TRANSLATE_OUT_OF_TIME:
            return FDErrorInfo(fd_code, "Translator hit the time limit")
        output = output.strip()
        if match := self.file_read_error.search(output):
            reason = match.group(2).strip()
            return FDErrorInfo(
                fd_code,
                f"Could not read file: {reason}",
                detect_file_type(output),
            )
        elif match := self.file_parse_error.search(output):
            file_type = match.group(1)
            reason = match.group(3).strip()
            return FDErrorInfo(
                fd_code,
                f"Could not parse {file_type} file: {reason}",
                detect_file_type(output),
            )
        elif match := self.domain_mismatch.search(output):
            task_domain = match.group(1)
            file_domain = match.group(2)
            return FDErrorInfo(
                fd_code,
                f"Domain name mismatch: task expects '{task_domain}' but domain file defines '{file_domain}'",
                detect_file_type(output),
            )
        elif match := self.derived_in_init.search(output):
            return FDErrorInfo(
                fd_code,
                f"Derived predicate '{match.group(1)}' appears in :init fact '{match.group(2)}'",
                detect_file_type(output),
            )
        elif match := self.derived_in_effect.search(output):
            return FDErrorInfo(
                fd_code,
                f"Derived predicate '{match.group(1)}' appears in effect of action '{match.group(2)}'",
                detect_file_type(output),
            )
        elif self.axiom_error.search(output):
            return FDErrorInfo(
                fd_code,
                "Axioms are not stratifiable",
                detect_file_type(output),
            )
        elif match := self.object_fluent.search(output):
            return FDErrorInfo(
                fd_code,
                f"Object fluents not supported: function '{match.group(1)}' has type '{match.group(2)}'",
                detect_file_type(output),
            )
        elif match := self.duplicate_object.findall(output):
            match_string = "\n".join(match)
            return FDErrorInfo(
                fd_code,
                f"Domain contains :constants referenced again in problem. Prefer object definition in problem only.{match_string}",
                PDDLFiles.DOMAIN,
            )
        return self._parse_semantic_error(output, fd_code)

    def _parse_semantic_error(self, output: str, fd_code: ExitCodes) -> FDErrorInfo:
        lines = output.split("\n")
        error_lines = []
        in_error = False
        skip_patterns = [
            lambda s: s == "Parsing...",
            lambda s: s.startswith("->") or s.startswith("\t->"),
            lambda s: s.startswith("Parsing ") and not in_error,
            lambda s: "[" in s and "wall-clock]" in s,
            lambda s: s.startswith("Translator "),
            lambda s: s.startswith("Generated "),
            lambda s: s.startswith("Computing "),
            lambda s: s.startswith("Building "),
            lambda s: s.startswith("Writing "),
            lambda s: s.startswith("Normalizing "),
            lambda s: s.startswith("Instantiating"),
            lambda s: s.startswith("Preparing "),
            lambda s: s.startswith("Completing "),
            lambda s: s.startswith("Detecting "),
            lambda s: s.startswith("Reordering "),
            lambda s: s.startswith("Processing "),
            lambda s: s.startswith("Simplifying "),
            lambda s: s.startswith("Collecting "),
            lambda s: s.startswith("Choosing "),
            lambda s: s.startswith("Finding "),
            lambda s: s.startswith("Checking "),
            lambda s: s == "Done!",
            lambda s: s.isdigit() or (s.replace(" ", "").replace(".", "").isdigit()),
            lambda s: s.startswith("INFO"),
            lambda s: s.startswith("Driver aborting"),
            lambda s: s.startswith("translate exit code"),
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # TODO: This could probably be implemented a lot cleaner
            should_skip = False
            for pattern in skip_patterns:
                if pattern(stripped):
                    should_skip = True
                    break
            if should_skip:
                continue
            in_error = True
            error_lines.append(stripped)
        # FIXME: Hitting this again
        assert len(error_lines) > 0
        return FDErrorInfo(
            fd_code,
            "\n".join(error_lines),
            detect_file_type(output),
        )


translate_parser = TranslateParser()
unsolvability_parser = UnsolvabilityParser()


def generate_plan(domain_file: Path, problem_file: Path) -> tuple[Path, ExitCodes, str]:
    plan_file = NamedTemporaryFile(delete=False)
    process = run(
        [
            "python",
            "../fast-downward-24.06.1/fast-downward.py",
            "--overall-time-limit",
            "1m",
            "--overall-memory-limit",
            "4G",
            "--plan-file",
            plan_file.name,
            "--alias",
            "seq-sat-lama-2011",
            domain_file,
            problem_file,
        ],
        capture_output=True,
        text=True,
    )
    fd_code = exit_codes[process.returncode]
    return Path(plan_file.name), fd_code, process.stdout


def translate_pddl(domain_file: Path, problem_file: Path) -> FDErrorInfo | None:
    """
    Executes the translation part of FD only.
    This is intended as a separate feedback step before plan-generation to find
    issues in the PDDL that could not be found using VAL.
    After this succceeds without issues, teh only issue that can still occur is
    the domain being unsolvable.
    """
    sas_file = NamedTemporaryFile(delete=False)
    process = run(
        [
            "python",
            "../fast-downward-24.06.1/src/translate/translate.py",
            "--sas-file",
            sas_file.name,
            domain_file,
            problem_file,
        ],
        capture_output=True,
        text=True,
    )
    fd_code = exit_codes[process.returncode]
    if fd_code != ExitCodes.SUCCESS:
        return translate_parser.parse_translate_error(process.stdout, fd_code)
    return None


def parse_error(
    planner_output: str,
    fd_code: ExitCodes,
    domain_file: Path,
    problem_file: Path,
    unsolvability_feedback: UnsolvabilityFeedback = UnsolvabilityFeedback.SIMPLE,
) -> FDErrorInfo:
    if is_translate_error(fd_code):
        return translate_parser.parse_translate_error(planner_output, fd_code)
    # TODO: We should also incorporate this information when successfully generating a plan
    #       as a last quality measure as well.
    elif is_unsolvable(fd_code):
        fallback = FDErrorInfo(
            fd_code,
            "Could not find a suitable plan",
        )
        if unsolvability_feedback == UnsolvabilityFeedback.FULL:
            return FDErrorInfo(
                fd_code,
                "Fast Downward could not find a plan. Following is the output from the planner\n\n"
                + planner_output,
            )
        elif unsolvability_feedback == UnsolvabilityFeedback.CURATED:
            return unsolvability_parser.parse_unsolvability_hints(
                planner_output, fd_code
            )
        elif unsolvability_feedback == UnsolvabilityFeedback.ABSTRACTION:
            predicates = AbstractionGenerator(
                domain_file, problem_file
            ).build_abstraction()
            if predicates:
                return FDErrorInfo(
                    fd_code,
                    "Fast Downward could not find a plan.\n\nHowever, trying to remove the following predicates from the domain and problem lead to a plan being found. "
                    + "While removing these predicates might be in conflict with the actual underlying task, there could be problems in your current definition/usage of them. "
                    + "Please look at the task and your current PDDL implementations again and try to fix the files so that fast downward can generate a plan that is applicable to the task.\n\n"
                    + "## Predicates:\n\n"
                    + predicates,
                )
            else:
                return fallback
        else:
            return fallback
    # Fall back which we should never really reach
    return FDErrorInfo(
        fd_code,
        "Fast Downward encountered an error while trying to generate a plan.",
    )
