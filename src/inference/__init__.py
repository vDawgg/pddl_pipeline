import os
from dataclasses import dataclass
from enum import StrEnum, auto


class Provider(StrEnum):
    LOCAL = auto()
    OPENAI = auto()
    OPENROUTER = auto()


@dataclass(frozen=True)
class ModelConfig:
    api_model_name: str
    provider: Provider

    @property
    def key_file(self) -> str:
        match self.provider:
            case Provider.OPENAI:
                return ".openai-key"
            case Provider.OPENROUTER:
                return ".openrouter-key"
            case Provider.LOCAL:
                return ".default-key"

    @property
    def base_url(self) -> str:
        match self.provider:
            case Provider.OPENAI:
                return "https://api.openai.com/v1"
            case Provider.OPENROUTER:
                return "https://openrouter.ai/api/v1"
            case Provider.LOCAL:
                return os.environ.get("SERVER_ADDRESS", "http://localhost:8080/v1")


class Models(StrEnum):
    QWEN_3_VL_8B = auto()
    QWEN_3_VL_235B = auto()
    QWEN_3_14B = auto()
    QWEN_25_CODER_14B = auto()
    GEMMA_3_12B = auto()
    QWEN_3_CODER = auto()
    QWEN_3_CODER_NEXT = auto()
    GPT_OSS = auto()
    GPT_52 = auto()
    GPT_5_NANO = auto()


MODEL_CONFIGS: dict[Models, ModelConfig] = {
    Models.QWEN_3_VL_8B: ModelConfig(
        api_model_name="qwen-3-vl",
        provider=Provider.LOCAL,
    ),
    Models.QWEN_3_VL_235B: ModelConfig(
        api_model_name="qwen/qwen3-vl-235b-a22b-instruct",
        provider=Provider.OPENROUTER,
    ),
    Models.QWEN_3_14B: ModelConfig(
        api_model_name="qwen-3",
        provider=Provider.LOCAL,
    ),
    Models.QWEN_25_CODER_14B: ModelConfig(
        api_model_name="qwen-2.5-coder",
        provider=Provider.LOCAL,
    ),
    Models.GEMMA_3_12B: ModelConfig(
        api_model_name="gemma3",
        provider=Provider.LOCAL,
    ),
    Models.QWEN_3_CODER: ModelConfig(
        api_model_name="qwen/qwen3-coder",
        provider=Provider.OPENROUTER,
    ),
    Models.QWEN_3_CODER_NEXT: ModelConfig(
        api_model_name="qwen/qwen3-coder-next",
        provider=Provider.OPENROUTER,
    ),
    Models.GPT_OSS: ModelConfig(
        api_model_name="gpt-oss",
        provider=Provider.LOCAL,
    ),
    Models.GPT_52: ModelConfig(
        api_model_name="gpt-5.2",
        provider=Provider.OPENAI,
    ),
    Models.GPT_5_NANO: ModelConfig(
        api_model_name="gpt-5-nano",
        provider=Provider.OPENAI,
    ),
}


def get_model_config(model: Models) -> ModelConfig:
    return MODEL_CONFIGS[model]
