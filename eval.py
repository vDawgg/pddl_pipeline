import logging
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.eval.eval_harness import run_eval
from src.utils.logger import configure_logging

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", default=1, type=int, required=False)
    parser.add_argument(
        "--pipeline",
        choices=[Pipelines.BASELINE, Pipelines.VAL_FEEDBACK],
        default=Pipelines.BASELINE,
        required=False,
    )
    parser.add_argument(
        "--model",
        choices=[Models.QWEN_3_VL_8B],
        default=Models.QWEN_3_VL_8B,
        required=False,
    )
    args = parser.parse_args()
    iterations = args.iterations
    pipeline = args.pipeline
    model = args.model

    configure_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    model_failures, syntax_domain, syntax_problem, failed_plans = run_eval(
        iterations, pipelines[pipeline](model)
    )
    logger.info("# Evaluation Results:")
    logger.info(f"Total number of iterations: {iterations}")
    logger.info(f"Number of generation failures: {model_failures}")
    logger.info(f"Number of syntactically incorrect domains: {syntax_domain}")
    logger.info(f"Number of syntactically incorrect problems: {syntax_problem}")
    logger.info(f"Number of unsolvable problems: {failed_plans}")
