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
- **apt-get in pods without network**: Pods without `hostNetwork: true` can't reach apt mirrors. Use `hostNetwork: true` + proxy: `export http_proxy=http://127.0.0.1:1083 https_proxy=http://127.0.0.1:1083` before `apt-get`
- **torchrun hostname resolution with hostNetwork**: K8s hostnames (e.g., `host-10-83-115-18`) may not resolve inside containers with `hostNetwork: true`. Fix: add `echo "127.0.0.1 $(hostname)" >> /etc/hosts` before torchrun
- **Image building on containerd-only clusters**: No docker; use `nerdctl run --net=host` + pip install + `nerdctl commit` to build images. Export with `ctr images export`, import with `ctr -n k8s.io images import`
- **Containerd proxy for image pulls**: Create `/etc/systemd/system/containerd.service.d/http-proxy.conf` with `HTTP_PROXY`/`HTTPS_PROXY` env vars, then `systemctl daemon-reload && systemctl restart containerd`
- **Production pods on GPU nodes**: Always check `kubectl get pods --all-namespaces -o wide | grep <node>` before deploying. dynamo-system pods use all 8 GPUs and must not be killed
- **Novita dataset has multiple export files**: The HuggingFace dataset `weilan55/novita20260309` has a different tar.gz than the local copy on .14. Always verify the correct file (799K records vs 11K) via `wc -l`

## Buffer Cleanup ↔ Training 协调 (Exp 13 Crash Postmortem)

**根因**: buffer_cleanup.py 每 60s 轮询一次，可以在 training DataLoader 正在加载文件时删除它们。Exp 13 Epoch 1 crash：cleanup 一次性删除全部 11045 个 train_count>=2 的文件，DataLoader 的所有 fallback 也失败。

**时间线**:
1. Epoch 0 结束 → `increment_train_count` 把 11045 文件标记为 train_count=2
2. Training 重读 manifest → 解析到 11045 文件（cleanup 还没跑）
3. Training 打印 "Epoch 1: 11045 train files"，开始构建 DataLoader
4. **buffer_cleanup 轮询** → 看到 11045 文件 train_count>=2 → **全部删除**
5. DataLoader workers 尝试加载 → FileNotFoundError → fallback 也失败 → crash

**修复** (3 层防护):
1. **Epoch lock file**: training 写 `.epoch_in_progress` 锁文件，cleanup 看到锁就跳过本轮
2. **`--max-delete-per-cycle`** (默认 5000): 每轮最多删 N 个文件，防止一次性清空
3. **`--min-retain-count`** (默认 1000): 始终保留至少 N 个文件在 manifest 中

**教训**:
- 后台清理进程 **永远不能** 假设数据没有被其他进程使用
- 删除操作必须有 rate limit 和 minimum retain 两个安全阀
- `data.py` 的 fallback 从 1 次增加到 5 次，但 fallback 不是主要防线

## 大规模数据同步原则

**问题**: 52K 文件 × 193MB = ~10TB 全量 rsync 耗时极长且无必要。Datagen 节点保存全部中间结果，但训练只需要最新的一个滚动窗口。

**原则**:
- **同步量要有上限**: 不要全量 rsync datagen 输出，只同步训练需要的量（如 ~1TB / ~5000 文件）
- **优先同步最新文件**: 旧文件已被训练过或多轮迭代后不再需要
- **Datagen 节点也需要清理**: datagen 生成无限输出，本地磁盘也需要 ring buffer
- **实操**: 停掉全量 rsync → 用 `--files-from` 只同步最新 N 个文件 → 后续用 `sync_datagen.sh` 增量同步 + `buffer_cleanup.py` 维持磁盘上限

**参考大小**:
| 场景 | 建议 buffer 上限 | 文件数 |
|------|-----------------|--------|
| 快速验证 | 200 GB | ~1000 |
| 标准训练 | 1 TB | ~5000 |
| 大规模训练 | 2 TB | ~10000 |
