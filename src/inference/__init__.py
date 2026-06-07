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
    QWEN_35_4B = auto()
    QWEN_35_122B = auto()
    GEMMA_3_12B = auto()
    GEMMA_4_E4B = auto()
    GEMMA_4_31B = auto()
    GEMINI_3_FLASH = auto()
    GPT_OSS_20B = auto()
    GPT_OSS_120B = auto()
    GPT_52 = auto()
    GPT_52_CODEX = auto()
    MINIMAX_M25 = auto()
    OPUS_45 = auto()
    OPUS_46 = auto()


MODEL_CONFIGS: dict[Models, ModelConfig] = {
    Models.GEMMA_4_E4B: ModelConfig(
        api_model_name="google/gemma-4",
        provider=Provider.LOCAL,
    ),
    Models.GEMMA_4_31B: ModelConfig(
        api_model_name="google/gemma-4",
        provider=Provider.LOCAL,
    ),
    Models.QWEN_35_4B: ModelConfig(
        api_model_name="qwen/qwen3.5-4b",
        provider=Provider.LOCAL,
    ),
    Models.QWEN_35_122B: ModelConfig(
        api_model_name="qwen/qwen3.5-122b-A10b",
        provider=Provider.LOCAL,
    ),
    Models.GPT_OSS_20B: ModelConfig(
        api_model_name="openai/gpt-oss-20b:nitro",
        provider=Provider.LOCAL,
    ),
    Models.GPT_OSS_120B: ModelConfig(
        api_model_name="openai/gpt-oss-120b:nitro",
        provider=Provider.LOCAL,
    ),
    Models.GEMINI_3_FLASH: ModelConfig(
        api_model_name="google/gemini-3-flash-preview",
        provider=Provider.OPENROUTER,
    ),
    Models.GPT_52: ModelConfig(
        api_model_name="gpt-5.2",
        provider=Provider.OPENAI,
    ),
    Models.GPT_52_CODEX: ModelConfig(
        api_model_name="gpt-5.2-codex",
        provider=Provider.OPENAI,
    ),
    Models.MINIMAX_M25: ModelConfig(
        api_model_name="minimax/minimax-m2.5:nitro",
        provider=Provider.OPENROUTER,
    ),
    Models.OPUS_45: ModelConfig(
        api_model_name="anthropic/claude-opus-4.5",
        provider=Provider.OPENROUTER,
    ),
    Models.OPUS_46: ModelConfig(
        api_model_name="anthropic/claude-opus-4.6",
        provider=Provider.OPENROUTER,
    ),
}


def get_model_config(model: Models) -> ModelConfig:
    return MODEL_CONFIGS[model]
