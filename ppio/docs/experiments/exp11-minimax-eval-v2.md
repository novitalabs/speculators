# Experiment 11: MiniMax-M2.5 Eagle3 Inference Evaluation v2 (Novita2 vs Aurora)

Evaluating the Novita2 (5K) draft model against Aurora-Spec and baseline.

- **Node**: .14 (TP=4, 4x H200)
- **Eval script**: `k8s/run_minimax_m2.5_eval.sh`
- **Config**: 10 prompts, 512 tokens each, temperature=0.6, ignore_eos=True, seed=42

## Eval 2: General prompts (coding + tech questions)

| Model | Tokens/s | Speedup | Acc Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-----------|-------|-------|-------|
| **Baseline** | 441.7 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 390.7 | 0.88x | 1.792 | 49.7% | 20.8% | 8.8% |
| **Novita2** (eager) | 113.4 | 0.26x | 1.268 | 25.6% | 1.2% | 0.1% |

## Eval 3: Novita in-domain prompts (from training data conversations)

| Model | Tokens/s | Speedup | Acc Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-----------|-------|-------|-------|
| **Baseline** | 426.7 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 376.7 | 0.88x | 1.596 | 40.1% | 14.6% | 4.8% |
| **Novita2** (eager) | 128.3 | 0.30x | 1.781 | **48.2%** | **20.9%** | **9.0%** |

## Eval 4: Held-out Novita prompts (weilan55/novita20260312_eval — not in training set, enforce_eager)

Data: 1354 conversations preprocessed from 44489 log records (March 12 export, not in training set).
Note: Novita2 still using `enforce_eager=True` (torch.compile cache conflict not yet fixed).

| Model | Tokens/s | Speedup | Acc Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-----------|-------|-------|-------|
| **Baseline** | 399.8 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 303.3 | 0.76x | 1.743 | **46.5%** | **19.0%** | **8.8%** |
| **Novita2** (eager) | 126.2 | 0.32x | 1.682 | 42.1% | 17.8% | 8.3% |

## Eval 5: torch.compile fix — Novita2 only (held-out 0312 data)

Fixed torch.compile by clearing vLLM compile cache (`~/.cache/vllm/torch_compile_cache/`) between eval runs.

| Model | Tokens/s (eager) | Tokens/s (compiled) | Speedup |
|-------|-----------------|--------------------:|---------|
| **Novita2** | 126.2 | **323.9** | **2.5x** |

## Eval 6: Full 3-way comparison with torch.compile fix (held-out 0312 data)

All models now using torch.compile (no enforce_eager). Cache cleared between each run.

| Model | Tokens/s | Speedup | Acc Length | Acc@0 | Acc@1 | Acc@2 |
|-------|----------|---------|-----------|-------|-------|-------|
| **Baseline** | 383.6 | 1.00x | - | - | - | - |
| **Aurora-Spec** | 386.6 | **1.01x** | 1.840 | **52.0%** | **22.7%** | **9.2%** |
| **Novita2** (compiled) | 320.4 | 0.84x | 1.668 | 40.9% | 17.3% | 8.6% |

## Cross-eval Summary

| Eval | Prompts | Aurora Acc@0 | Novita2 Acc@0 | Winner |
|------|---------|-------------|---------------|--------|
| Eval 2 | General (coding/tech) | 49.7% | 25.6% | Aurora |
| Eval 3 | Novita in-domain (0309) | 40.1% | **48.2%** | **Novita2** |
| Eval 4-6 | Novita held-out (0312) | **52.0%** | 40.9% | Aurora |

## torch.compile Cache Conflict — Root Cause and Fix

**Problem**: Running different draft models sequentially in the same pod caused `AssertionError: expected size 2048==1280` during `profile_run`.

**Root cause**: vLLM's compile cache at `~/.cache/vllm/torch_compile_cache/<hash>/rank_X_Y/eagle_head` uses a hash based on the target model config, not the draft model config. When Aurora (num_attention_heads=24, QKV dim=1280/TP) and Novita2 (num_attention_heads=48, QKV dim=2048/TP) run sequentially, the second eval reuses the first's cached compiled graph with incompatible tensor shapes.

**Fix**: Clear `~/.cache/vllm/torch_compile_cache/` before each eval run:
```bash
rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
```

**Impact**: Novita2 throughput improved from 126 tok/s (enforce_eager) to 320-324 tok/s (compiled) — **2.5× speedup**.

## Analysis

1. **Aurora still wins overall**: 52% acceptance and 1.01x throughput (slight speedup over baseline). Novita2 at 41% acceptance is 0.84x baseline — overhead still exceeds the speculation benefit.

2. **torch.compile fix eliminates the throughput penalty**: Novita2 went from 0.32x to 0.84x baseline. The remaining gap is purely due to lower acceptance rates, not compilation issues.

3. **Acceptance rate is the bottleneck**: At 41% Acc@0 with 3 draft tokens, the average acceptance length of 1.67 is not enough to overcome the draft model's overhead. Need ~50%+ Acc@0 to break even.

4. **Aurora generalizes better**: Despite being trained for MiniMax-M2.1 (not M2.5), Aurora's training on diverse data gives it better generalization than our domain-specific 5K Novita model.

## Key Takeaways

- **torch.compile cache conflict was the #1 throughput blocker** — now fixed
- Domain-specific Novita training data helps on in-distribution eval but doesn't generalize
- To beat Aurora, need: (a) more diverse training data (50K+ mixed domain), (b) higher acceptance rates (train longer or with more data), or (c) match Aurora's architecture (num_attention_heads=24) for potential compile optimization benefits

- **Status**: COMPLETED
