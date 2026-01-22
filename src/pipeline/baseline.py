import logging
from pathlib import Path
from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from src.base.pipeline import PipelineBase, Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.constants import project_root
from src.eval.fast_downward import FDErrorInfo
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference import Models, provider_hosts, provider_keys
from src.inference.model_comm import (
    make_assistant_message,
    make_user_message,
    make_user_message_with_image,
)
from src.utils.domains import Domains
from src.utils.prompts import domain_pompts, problem_prompts

logger = logging.getLogger(__name__)


class Baseline(PipelineBase):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.BASELINE)

    def is_domain_valid(self, domain_file: Path) -> bool:
        err_info = get_syntax_mistakes_domain(domain_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid domain")
            return False
        logger.debug("Generated syntactically valid domain")
        return True

    def is_problem_valid(
        self,
        domain_file: Path,
        problem_file: Path,
    ) -> bool:
        err_info = get_syntax_mistakes_problem(domain_file, problem_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid problem")
            return False
        logger.debug("Generated syntactically valid problem")
        return True

    T = TypeVar("T", bound=BaseModel)

    def make_request[T](
        self,
        input_prompt: str,
        messages: list[Any] | None = None,
        format: type[T] | None = None,
        imgs: list[str] | None = None,
    ) -> tuple[T | str, list[Any]]:
        messages = messages or []

        logger.debug("# User Message")
        logger.debug(input_prompt)

        key_path = project_root / provider_keys[self.model]
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")

        client = openai.OpenAI(
            base_url=provider_hosts[self.model],
            api_key=open(str(key_path)).readline().strip(),
        )

        if imgs:
            messages.append(make_user_message_with_image(input_prompt, imgs))
        else:
            messages.append(make_user_message(input_prompt))

        if format:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=format,
            )
            res = response.choices[0].message.parsed
            assert res
            return res, messages
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            res = response.choices[0].message.content
            logger.debug("# Assistant Message")
            logger.debug(res)
            assert res is not None
            res = res.removeprefix("```pddl")
            res = res.removesuffix("```")
            return res.strip(), messages

    def _run_impl(self) -> PipelineResult:
        domain, messages = self.make_request(
            domain_pompts[self.domain],
        )
        self.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        if not self.is_domain_valid(self.domain_file):
            return self.create_result(error=PipelineError.DOMAIN_FAILURE)

        problem, messages = self.make_request(
            problem_prompts[self.domain],
            messages=[*messages, make_assistant_message(domain)],
        )
        self.problem_file = self._write_pddl_file(
            problem, pddl_file_type=PDDLFiles.PROBLEM
        )
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            return self.create_result(error=PipelineError.PROBLEM_FAILURE)

        planner_output = self._generate_plan(self.domain_file, self.problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(error=planner_output.to_pipeline_error())
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result()
