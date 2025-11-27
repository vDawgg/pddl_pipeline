from src.inference.model_comm import make_request
from src.pipeline.pipeline_base import PipelineBase
from src.utils.prompts import Prompts, get_prompt


class Baseline(PipelineBase):
    def run(self) -> tuple[str, str]:
        # TODO: We should add natural language descriptions of the actions we pass to the model
        # TODO: Think about whether we want to fail early here and in the pipelines in general if we fail
        #       to generate a valid domain so we can skip having to create a problem file
        domain, messages = make_request(
            get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN),
            self.model,
        )
        problem, messages = make_request(
            get_prompt(Prompts.BASELINE_PROBLEM),
            self.model,
            messages=messages,
        )
        return domain, problem
