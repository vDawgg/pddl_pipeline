import logging
from argparse import ArgumentParser

from src.base.pipeline import Tools
from src.base.schemas import Domains, Problems
from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.utils.logger import configure_logging

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--pipeline",
        choices=[pipeline.value for pipeline in Pipelines],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=[model.value for model in Models],
        required=True,
    )
    parser.add_argument(
        "--domain",
        choices=[domain.value for domain in Domains],
        required=True,
    )
    parser.add_argument(
        "--problem",
        choices=[problem.value for problem in Problems],
        required=True,
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
    problem = args.problem
    ablate_tools_args = args.ablate_tools
    ablate_tools = []
    if ablate_tools_args is not None:
        for tool in str(ablate_tools_args).split(","):
            try:
                ablate_tools.append(Tools(tool))
            except Exception:
                raise ValueError(
                    f"Specified tool '{tool}' for ablation does not match any known tool."
                ) from None
    optimize = args.optimize
    optimized_program = args.optimized_program

    configure_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    optimizable = [
        Pipelines.TOOL_CALL,
        Pipelines.TOOL_CALL_ABSTRACTION,
        Pipelines.TOOL_CALL_CURATED,
        Pipelines.TOOL_CALL_FULL,
        Pipelines.RIGID_TRAJECTORY,
    ]
    if optimize and pipeline in optimizable:
        pipelines[pipeline](model, domain, problem).compile_module()
    else:
        results_file = pipelines[pipeline](
            model, domain, problem, ablate_tools, optimized_program=optimized_program
        ).run_eval(iterations)
        logger.info(f"# Saved results to {results_file}")
