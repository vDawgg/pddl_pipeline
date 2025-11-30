import logging
from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.logger import configure_logging


if __name__ == "__main__":
    parser = ArgumentParser()
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
    pipeline = args.pipeline
    model = args.model

    configure_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)

    pipelines[pipeline](model).run()
