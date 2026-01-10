import logging
import os
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.eval.eval_harness import run_eval
from src.utils.logger import configure_logging
from src.utils.domains import Domains

# TODO: For ease of use this should probably just be merged with main.py
if __name__ == "__main__":
    env_iterations = os.environ.get("EVAL_ITERATIONS", 1)
    env_pipeline = os.environ.get("EVAL_PIPELINE", Pipelines.BASELINE.value)
    env_model = os.environ.get("EVAL_MODEL", Models.QWEN_3_VL_8B.value)
    env_domain = os.environ.get("EVAL_DOMAIN", Domains.BLOCKSWORLD.value)

    parser = ArgumentParser()
    parser.add_argument(
        "--iterations", default=env_iterations, type=int, required=False
    )
    parser.add_argument(
        "--pipeline",
        choices=[pipeline.value for pipeline in Pipelines],
        default=env_pipeline,
        required=False,
    )
    parser.add_argument(
        "--model",
        choices=[model.value for model in Models],
        default=env_model,
        required=False,
    )
    parser.add_argument(
        "--domain",
        choices=[domain.value for domain in Domains],
        default=env_domain,
        required=False,
    )
    args = parser.parse_args()
    iterations = args.iterations
    pipeline = args.pipeline
    model = args.model
    domain = args.domain

    configure_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    eval_results = run_eval(iterations, pipelines[pipeline](model, domain))
    logger.info("# Evaluation Results:")
    logger.info(f"Total number of iterations: {iterations}")
    logger.info(f"Number of generation failures: {eval_results.num_model_errors}")
    logger.info(
        f"Number of syntactically incorrect domains: {eval_results.num_syntactically_incorrect_domains}"
    )
    logger.info(
        f"Number of syntactically incorrect problems: {eval_results.num_syntactically_incorrect_problems}"
    )
    logger.info(f"Number of unsolvable problems: {eval_results.num_failed_plans}")
    logger.info(
        f"Total number of feedback loops: {eval_results.get_all_total_iterations()}"
    )
    logger.info(
        f"Average number of feedback loops: {eval_results.get_all_avg_iterations()}"
    )
