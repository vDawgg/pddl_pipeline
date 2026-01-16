import os
from collections import defaultdict
from enum import StrEnum


class Models(StrEnum):
    QWEN_3_VL_8B = "qwen-3-vl"
    QWEN_25_CODER_14B = "qwen-2.5-coder"
    GEMMA_3_12B = "gemma3"
    GPT_52 = "gpt-5.2"
    GPT_5_NANO = "gpt-5-nano"


# TODO: We need to think of something better for this
provider_keys = defaultdict(
    lambda: ".default-key",
)
provider_keys[Models.GPT_52] = ".openai-key"
provider_keys[Models.GPT_5_NANO] = ".openai-key"

provider_hosts = defaultdict(
    lambda: os.environ.get("SERVER_ADDRESS", "http://localhost:8080/v1")
)
provider_hosts[Models.GPT_52] = "https://api.openai.com/v1"
provider_hosts[Models.GPT_5_NANO] = "https://api.openai.com/v1"
