"""HF-compat shims for verifier architectures newer than the pinned transformers.

The speculators image pins transformers 4.57.x, which predates GLM-5.2's
``glm_moe_dsa`` model type. vLLM 0.17 already registers GlmMoeDsaForCausalLM
but resolves the HF config through ``AutoConfig``, so both camelot and the
in-process vLLM engine fail on config load without this registration. The
config class lives here (importable from every process, including vLLM
workers that unpickle VllmConfig.hf_config) rather than in camelot.
"""

from transformers import AutoConfig, PretrainedConfig


class GlmMoeDsaConfig(PretrainedConfig):
    # Minimal shim: PretrainedConfig.from_dict sets every config.json field as
    # an attribute, which is all camelot/vLLM read (num_hidden_layers,
    # hidden_size, vocab_size, kv_lora_rank, ...).
    model_type = "glm_moe_dsa"


def register_glm_moe_dsa() -> None:
    try:
        AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig)
    except ValueError:
        # Already registered (idempotent across entrypoints).
        pass
