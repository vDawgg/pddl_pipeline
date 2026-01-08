import os
from enum import StrEnum
from collections import defaultdict


class Models(StrEnum):
    QWEN_3_VL_8B = "qwen-3-vl"
    QWEN_25_CODER_14B = "qwen-2.5-coder"
    GPT_52 = "gpt-5.2"


provider_keys = defaultdict(
    lambda: ".default-key",
)
provider_keys[Models.GPT_52] = ".openai-key"

provider_hosts = defaultdict(
    lambda: os.environ.get("SERVER_ADDRESS", "http://localhost:8080/v1")
)
provider_hosts[Models.GPT_52] = "https://api.openai.com/v1"
