# Experiment 9: MiniMax-M2.5 Eagle3 Inference Evaluation (vLLM Spec Decode)

Evaluating Eagle3 speculative decoding speedup for MiniMax-M2.5 with different draft models.

- **Node**: .14 (8x H200 143GB, using 4 GPUs via TP=4)
- **Image**: speculators:v0.17.0 (vLLM 0.17.0)
- **Base model**: MiniMax-M2.5 (FP8 quantized)
- **Draft models tested**:
  1. **Aurora-Spec**: `togethercomputer/Aurora-Spec-Minimax-M2.1` (1.6GB, `LlamaForCausalLMEagle3`, draft_vocab=32000)
  2. **Novita-trained**: Our Exp 7 model converted to vLLM format (2.7GB, `LlamaForCausalLMEagle3`, draft_vocab=200064)
- **Config**:
  - TP=4, max_model_len=8192, gpu_memory_utilization=0.85
  - 10 prompts (coding + general), 512 tokens each, temperature=0.6, ignore_eos=True
  - num_speculative_tokens=3 for both draft models
- **Eval script**: `k8s/run_minimax_m2.5_eval.sh`
- **K8s**: `k8s/k8s-minimax-m2.5-eval.yaml`
- **Converter**: `k8s/convert_speculators_to_vllm_eagle3.py`

## vLLM Patching Required

vLLM 0.17.0 does not natively support Eagle3 speculative decoding for `minimax_m2` model type. Created `k8s/patch_vllm_minimax_eagle3.py` which applies two patches at container startup:

1. **Whitelist patch** (`vllm/config/speculative.py`): Adds `minimax_m2` to Eagle3 supported model types
2. **Interface patch** (`vllm/model_executor/models/minimax_m2.py`): Adds `SupportsEagle3` protocol to `MiniMaxM2ForCausalLM`:
   - `supports_eagle3 = True` class variable
   - `has_own_lm_head = False` and `has_own_embed_tokens = False` (required by `SupportsEagleBase` protocol)
   - `set_aux_hidden_state_layers()` and `get_eagle3_aux_hidden_state_layers()` methods
   - Modified `MiniMaxM2Model.forward()` to collect auxiliary hidden states at specified layers
   - Auxiliary layers: `(2, num_layers // 2, num_layers - 3)` = `(2, 31, 59)` for 62-layer model

## Speculators → vLLM Model Conversion

Created `k8s/convert_speculators_to_vllm_eagle3.py` to convert speculators `Eagle3DraftModel` format to vLLM `Eagle3LlamaForCausalLM` format:

1. **Config**: Translates speculators `transformer_layer_config` → standard `LlamaConfig` with `architectures: ["LlamaForCausalLMEagle3"]`
2. **Weight key renaming**: `layers.0.*` → `midlayer.*` (vLLM's `load_weights` reverses this back to `layers.0.*`)
3. **Vocab mapping**: Adds identity `d2t` (draft-to-target) mapping when `draft_vocab_size == target_vocab_size`; vLLM expects weight name `d2t` not `draft_id_to_target_id`
4. **Dtype**: Converts fp32 → bfloat16

Weight verification confirmed all weights match exactly after conversion (max_diff=0).

## Issues Encountered and Fixed

1. **vLLM forces `spawn` multiprocessing**: When CUDA is initialized, vLLM overrides `VLLM_WORKER_MULTIPROC_METHOD=fork` to `spawn`. All patches must be on-disk `.py` files, not runtime monkey-patches.
   - **Fix**: `PYTHONDONTWRITEBYTECODE=1` + clearing `__pycache__` ensures spawned workers use patched files

2. **`@runtime_checkable Protocol` isinstance check**: `isinstance(model, SupportsEagle3)` checks ALL protocol attributes including inherited `has_own_lm_head` and `has_own_embed_tokens` from `SupportsEagleBase`. Missing attributes caused the check to fail silently.
   - **Fix**: Added `has_own_lm_head = False` and `has_own_embed_tokens = False` class variables to `MiniMaxM2ForCausalLM`

3. **Speculators format incompatible with vLLM**: Speculators saves as `Eagle3DraftModel` with `layers.0.*` weight prefix. vLLM expects `LlamaForCausalLMEagle3`/`Eagle3LlamaForCausalLM` architecture with `midlayer.*` prefix and standard LlamaConfig.
   - **Fix**: Created conversion script `k8s/convert_speculators_to_vllm_eagle3.py`

4. **torch.compile shape mismatch with non-Aurora draft models**: `AssertionError: expected size 2048==1280` during `profile_run`. The Aurora model has `num_attention_heads=24` (QKV per TP shard = 1280), but our Novita model has `num_attention_heads=48` (QKV per TP shard = 2048). The compiled graph from one draft model cannot be reused for another with different attention configs. Per-eval `TORCHINDUCTOR_CACHE_DIR` isolation did not fix it — the issue is in vLLM's own compile cache, not torchinductor.
   - **Workaround**: Use `enforce_eager=True` for the Novita model to disable torch.compile entirely. This makes the comparison unfair (eager mode is slower), so throughput numbers are not directly comparable.
   - **Root cause**: Likely vLLM's compile cache at `~/.cache/vllm/` persists across subprocesses in the same pod, and the cache key doesn't differentiate draft model attention configs

5. **Aux hidden state capture timing**: Initial patch captured hidden states BEFORE the layer executed (`hidden_states + residual` before `layer()`), but speculators data generation captures AFTER (`layer()` then `hidden_states + residual`). Fixed to capture after layer execution, but this did not improve acceptance rates — the off-by-one had minimal impact.

## Results

| Model | Tokens/s | Speedup | Acceptance Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-------------------|-------|-------|-------|
| **Baseline** (no spec) | 436.9 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 509.6 | **1.17x** | 1.803 | 52.15% | 24.44% | 12.0% |
| **Novita-trained** (eager) | 108.6 | 0.25x | 1.255 | 24.24% | 1.2% | 0.02% |

- **Aurora-Spec**: 17% throughput improvement with mean acceptance length of 1.80 tokens
  - Acceptance rates: ~52% for 1st draft token, ~24% for 2nd, ~12% for 3rd
  - Decent performance considering it was trained for MiniMax-M2.1, not M2.5
- **Novita-trained**: Very poor acceptance rates (24% at position 0)
  - Throughput is 0.25x baseline (slower due to `enforce_eager` + draft overhead)
  - The model had 81.5% top-1 accuracy on Novita val data, but only 24% on general prompts
  - Cause: severe overfitting on 812 Novita samples (coding agent conversations)
  - The model learned to predict tokens in multi-turn coding agent conversations, but doesn't generalize to single-turn general/coding prompts

## Analysis

1. **Aurora-Spec is the better choice** for general MiniMax-M2.5 speculative decoding. Despite being trained for M2.1, it generalizes well to M2.5 and provides a real 17% speedup.

2. **Our Novita-trained model needs more diverse training data**. Training on only 812 domain-specific samples caused severe overfitting. To match or beat Aurora-Spec, we need:
   - 5000+ diverse samples (sharegpt + ultrachat + novita)
   - The Experiment 8 (50K online data) model may perform much better

3. **torch.compile compatibility**: Draft models with different `num_attention_heads` than Aurora (24) cause compile cache conflicts. This is a vLLM limitation that requires `enforce_eager=True` as a workaround, which significantly hurts throughput.

4. **MiniMax-M2.5 Eagle3 architecture notes**:
   - `hidden_size=3072`, `num_attention_heads=48`, `head_dim=128` → Q dimension (6144) = 2× hidden_size
   - Aurora uses `num_attention_heads=24` making Q = hidden_size (standard)
   - Eagle3 draft model's QKV input is always `2 × hidden_size` (embeds + hidden concatenated)
   - `fc` layer: `3 × hidden_size → hidden_size` (3 aux hidden states from layers 2, 31, 59)

- **Status**: COMPLETED
