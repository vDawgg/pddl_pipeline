import logging
import os
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.domains import Domains
from src.utils.logger import configure_logging

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

    results_file = pipelines[pipeline](model, domain).run_eval(iterations)
    logger.info(f"# Saved results to {results_file}")
