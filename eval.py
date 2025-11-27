from argparse import ArgumentParser

from src.inference import Models
from src.pipeline import Pipelines, pipelines
from src.eval.eval_harness import run_eval

if __name__ == "__main__":
    # TODO: Maybe include the model that should be used as parser arg
    #       Later on this should also include the domain to be generated
    parser = ArgumentParser()
    parser.add_argument("--iterations", default=1, type=int, required=False)
    parser.add_argument(
        "--pipeline",
        choices=[Pipelines.BASELINE],
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

    syntax_domain, syntax_problem, plans = run_eval(
        iterations, pipelines[pipeline](model)
    )
    print("# Evaluation Results:")
    print("Total number of iterations:", iterations)
    print("Number of syntactically incorrect domains:", syntax_domain)
    print("Number of syntactically incorrect problems:", syntax_problem)
    print("Number of unsolvable problems:", plans)
