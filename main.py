import logging
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.domains import Domains
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
        default=Domains.BLOCKSWORLD,
        required=False,
    )
    args = parser.parse_args()
    pipeline = args.pipeline
    model = args.model
    domain = args.domain

    configure_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)

    pipelines[pipeline](model, domain).run()
