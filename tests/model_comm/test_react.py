from src.inference.model_comm import (
    ReactResponse,
    make_prompt_with_trajectory,
    parse_react_message,
)
from tests.constants import model_comm_resource_dir


class TestReact:
    def test_parse_react_message_normal(self):
        with open(model_comm_resource_dir / "react_message_normal.txt") as f:
            parsed = parse_react_message(f.read())
            assert parsed is not None
            assert parsed.thought == "Some thought"
            assert parsed.tool_name == "some_tool"
            assert parsed.tool_args == {"content": "something"}

    def test_parse_react_message_additional_unwanted_fields(self):
        with open(
            model_comm_resource_dir / "react_message_additional_unwanted_fields.txt"
        ) as f:
            parsed = parse_react_message(f.read())
            assert parsed is not None
            assert parsed.thought == "Some thought"
            assert parsed.tool_name == "some_tool"
            assert parsed.tool_args == {"content": "something"}

    def test_parse_react_message_no_match(self):
        with open(model_comm_resource_dir / "react_message_no_match.txt") as f:
            parsed = parse_react_message(f.read())
            assert parsed is None

    def test_make_prompt_with_trajectory(self):
        with open(
            model_comm_resource_dir / "prompt_with_two_iteration_trajectory.txt"
        ) as f:
            input_prompt = "Some prompt for the model to follow\n"
            parsed_responses = [
                ReactResponse(
                    thought="First thought",
                    tool_name="first_tool",
                    tool_args={"first_arg": 1},
                ),
                ReactResponse(
                    thought="Second thought",
                    tool_name="second_tool",
                    tool_args={"second_arg": 2},
                ),
            ]
            results = ["result_one", "result_two"]
            prompt_with_trajectory = make_prompt_with_trajectory(
                input_prompt, parsed_responses, results
            ).strip()
            assert prompt_with_trajectory == f.read().strip()
