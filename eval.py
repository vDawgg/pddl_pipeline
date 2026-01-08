import logging
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.eval.eval_harness import run_eval
from src.utils.logger import configure_logging
from src.utils.domains import Domains

# TODO: For ease of use this should probably just be merged with main.py
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", default=1, type=int, required=False)
    parser.add_argument(
        "--pipeline",
        choices=[pipeline.value for pipeline in Pipelines],
        default=Pipelines.BASELINE,
        required=False,
    )
    parser.add_argument(
        "--model",
        choices=[model.value for model in Models],
        default=Models.QWEN_3_VL_8B,
        required=False,
    )
    parser.add_argument(
        "--domain",
        choices=[domain.value for domain in Domains],
        default=Domains.BLOCKSWORLD,
        required=False,
    )
    args = parser.parse_args()
    iterations = args.iterations
    pipeline = args.pipeline
    model = args.model
    domain = args.domain

    configure_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    model_failures, syntax_domain, syntax_problem, failed_plans = run_eval(
        iterations, pipelines[pipeline](model, domain)
    )
    logger.info("# Evaluation Results:")
    logger.info(f"Total number of iterations: {iterations}")
    logger.info(f"Number of generation failures: {model_failures}")
    logger.info(f"Number of syntactically incorrect domains: {syntax_domain}")
    logger.info(f"Number of syntactically incorrect problems: {syntax_problem}")
    logger.info(f"Number of unsolvable problems: {failed_plans}")
