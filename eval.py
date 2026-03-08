import logging
import os
from argparse import ArgumentParser

from src.base.pipeline import Tools
from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.domains import Domains
from src.utils.logger import configure_logging

# TODO: For ease of use this should probably just be merged with main.py
if __name__ == "__main__":
    env_iterations = os.environ.get("EVAL_ITERATIONS", 1)
    env_pipeline = os.environ.get("EVAL_PIPELINE", Pipelines.RIGID_TRAJECTORY.value)
    env_model = os.environ.get("EVAL_MODEL", Models.QWEN_3_VL_8B.value)
    env_domain = os.environ.get("EVAL_DOMAIN", Domains.RING_AND_PEG.value)

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
    parser.add_argument(
        "--ablate_tools",
        help=f"Tools to able from the tool-calling pipeline. Should be given as comma separated list. Available tools are: [{[tool.value for tool in Tools]}]",
        required=False,
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
    )
    parser.add_argument(
        "--optimized_program",
        required=False,
    )
    args = parser.parse_args()
    iterations = args.iterations
    pipeline = args.pipeline
    model = args.model
    domain = args.domain
    ablate_tools = str(args.ablate_tools).split(",")
    optimize = args.optimize
    optimized_program = args.optimized_program

    configure_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    if optimize and pipeline in [Pipelines.TOOL_CALL, Pipelines.RIGID_TRAJECTORY]:
        pipelines[pipeline](model, domain).compile_module()
    else:
        results_file = pipelines[pipeline](
            model, domain, ablate_tools, optimized_program=optimized_program
        ).run_eval(iterations)
        logger.info(f"# Saved results to {results_file}")
