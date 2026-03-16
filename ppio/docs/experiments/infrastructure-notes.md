# Infrastructure Notes

- **CUDA compat lib conflict**: Must set `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:...` in pod env
- **Data gen segfault on TP=8**: Use TP=2 (Qwen3-32B) or TP=4 (MiniMax-M2.5)
- **CNI bridge conflict on .18**: Fixed by deleting stale cni0 (`ip link delete cni0`)
- **No direct network in pods**: Pre-download data, set `HF_HUB_OFFLINE=1`
- **Proxy on .18**: `https_proxy=http://127.0.0.1:1083` for HuggingFace access
- **MiniMax-M2.5 requires vLLM >= 0.17.0**: v0.16.0 has no MiniMaxM2 model support
- **trust_remote_code for custom models**: Must add `trust_remote_code=True` to all `AutoConfig.from_pretrained` calls
- **NaN data from MoE models**: MiniMax-M2.5 occasionally produces NaN hidden states (~0.4%). Always validate generated data before training
- **shm sizing for MoE**: Large MoE models need 256Gi+ shared memory to avoid OOM with DataLoader workers
- **DeepGEMM warmup**: MoE models need ~4 min first-time kernel compilation on new nodes. Looks like a hang but is normal
- **Online training `_attn_implementation`**: `train_streaming.py` must set `transformer_layer_config._attn_implementation = "simple_flex_attention"` — this is a `classmethod` registration on transformers' global `AttentionInterface._global_mapping`
- **Multi-node rsync**: Can't rsync between two remote hosts directly; SSH to source and rsync from there
- **Eagle3 requires full model weights on training node**: Not just config — the embedding layer is loaded from target model safetensors
- **NCCL timeout for large-scale FSDP training**: Default 600s timeout is insufficient for validation/checkpoint with large val sets on MoE models. Set `NCCL_TIMEOUT=3600` env var (read by `utils.py:maybe_setup_distributed`). Use `--max-val-files 200` to cap validation set size
- **rsync in vllm containers**: vllm-openai images don't include rsync/ssh-client. Must `apt-get install rsync openssh-client` after pod start
- **Bash glob limit**: `ls *.pt` fails with 30K+ files. Use `find -name '*.pt' | wc -l` instead
- **Code sync to pod nodes**: K8s pods with hostPath mounts use the node's local filesystem. Always `rsync` code changes to the target node before restarting pods
