import logging
from argparse import ArgumentParser

from src.base.pipeline import Tools
from src.base.schemas import Domains, Problems
from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.logger import configure_logging

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=[pipeline.value for pipeline in Pipelines],
        default=Pipelines.RIGID_TRAJECTORY,
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
        default=Domains.RING_AND_PEG,
        required=False,
    )
    parser.add_argument(
        "--problem",
        choices=[problem.value for problem in Problems],
        default=Problems.RING_AND_PEG_1,
        required=False,
    )
    parser.add_argument(
        "--ablate_tools",
        help=f"Tools to able from the tool-calling pipeline. Should be given as comma separated list. Available tools are: [{[tool.value for tool in Tools]}]",
        required=False,
    )
    args = parser.parse_args()
    pipeline = args.pipeline
    model = args.model
    domain = args.domain
    problem = args.problem
    ablate_tools = args.ablate_tools
    if args.ablate_tools:
        ablate_tools = "".split(",")

    configure_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)

    pipelines[pipeline](model, domain, problem, ablate_tools).run()
