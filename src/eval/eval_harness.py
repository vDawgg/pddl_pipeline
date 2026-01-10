import logging
from dataclasses import dataclass, field

from tqdm import tqdm

from src.base.pipeline import PipelineBase
from src.base.schema import PipelineError


logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    num_model_errors: int = 0
    num_syntactically_incorrect_domains: int = 0
    num_syntactically_incorrect_problems: int = 0
    num_failed_plans: int = 0
    num_successful_plans: int = 0
    iteration_counts: list[dict[str, int]] = field(default_factory=list)

    def get_avg_iterations(self, key: str) -> float:
        counts = [d.get(key, 0) for d in self.iteration_counts if key in d]
        return sum(counts) / len(counts) if counts else 0.0

    def get_total_iterations(self, key: str) -> int:
        return sum(d.get(key, 0) for d in self.iteration_counts)

    def get_iteration_keys(self) -> set[str]:
        keys: set[str] = set()
        for d in self.iteration_counts:
            keys.update(d.keys())
        return keys

    def get_all_avg_iterations(self) -> dict[str, float]:
        d = {}
        for k in self.get_iteration_keys():
            d[k] = self.get_avg_iterations(k)
        return d

    def get_all_total_iterations(self) -> dict[str, int]:
        d = {}
        for k in self.get_iteration_keys():
            d[k] = self.get_total_iterations(k)
        return d


# TODO: This function will need to persist results so nothing gets lost
#       → Does not need to be the full PDDL results
def run_eval(iterations: int, pipeline: PipelineBase) -> EvalResults:
    results = EvalResults()

    for _ in tqdm(range(iterations), "Running Evaluation"):
        try:
            result = pipeline.run()
            results.iteration_counts.append(result.iterations)

            match result.error:
                case PipelineError.DOMAIN_FAILURE:
                    results.num_syntactically_incorrect_domains += 1
                case PipelineError.PROBLEM_FAILURE:
                    results.num_syntactically_incorrect_problems += 1
                case PipelineError.PLAN_FAILURE:
                    results.num_failed_plans += 1
                case None:
                    results.num_successful_plans += 1
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results.num_model_errors += 1
            continue

    return results
