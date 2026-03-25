#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Evaluation on ZClawBench Agent Prompts
# Compare: baseline, Aurora-Spec-M2.1, Exp14 ckpt54, Exp15 ckpt67
# Tests speculative decoding speedup on real agent task prompts
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

BASE_MODEL="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
AURORA_SPEC="${AURORA_SPEC:-/data/models/Aurora-Spec-Minimax-M2.1}"
EXP14_SPEC="${EXP14_SPEC:-/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints/54}"
EXP15_SPEC="${EXP15_SPEC:-/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67}"
OUTPUT_DIR="/data/output/minimax_m2.5_eval_zclawbench"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

mkdir -p "$OUTPUT_DIR"

cd /workspace/speculators

###############################################################################
# Patch vLLM
###############################################################################
python3 /workspace/speculators/k8s/patch_vllm_minimax_eagle3.py
find /usr/local/lib/python3.12/dist-packages/vllm/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

###############################################################################
# Write the evaluation Python script
###############################################################################
cat > "$OUTPUT_DIR/eval_runner.py" << 'PYEOF'
import json
import os
import sys
import time
import gc

BASE_MODEL = os.environ["BASE_MODEL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
TP = int(os.environ["TP"])
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"])
ZCLAWBENCH_PATH = os.environ.get("ZCLAWBENCH_PATH", "/workspace/speculators/k8s/zclawbench_prompts.json")

# Load ZClawBench prompts
with open(ZCLAWBENCH_PATH) as f:
    all_prompts = json.load(f)

# Convert to chat format for vLLM
CHAT_PROMPTS = [p["messages"] for p in all_prompts]
PROMPT_CATEGORIES = [p["category"] for p in all_prompts]
print(f"[INFO] Loaded {len(CHAT_PROMPTS)} ZClawBench prompts")
from collections import Counter
cat_counts = Counter(PROMPT_CATEGORIES)
for cat, cnt in cat_counts.most_common():
    print(f"  {cat}: {cnt}")


def run_eval(name, spec_config=None, enforce_eager=False):
    """Run a single evaluation and return metrics dict."""
    from vllm import LLM, SamplingParams

    result_file = os.path.join(OUTPUT_DIR, f"{name}_results.json")
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")

    llm_kwargs = {
        "model": BASE_MODEL,
        "tensor_parallel_size": TP,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.85,
        "trust_remote_code": True,
        "disable_log_stats": False,
        "seed": 42,
    }
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if spec_config:
        llm_kwargs["speculative_config"] = spec_config

    print(f"[INFO] LLM kwargs: {json.dumps({k: str(v) for k, v in llm_kwargs.items()}, indent=2)}")

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to load model for {name}: {e}")
        import traceback; traceback.print_exc()
        error_result = {"name": name, "error": str(e)}
        with open(result_file, "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result

    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=512, ignore_eos=True,
    )

    print(f"[INFO] Running chat inference on {len(CHAT_PROMPTS)} ZClawBench prompts (512 tokens each)...")
    start = time.time()
    outputs = llm.chat(CHAT_PROMPTS, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    metrics = {
        "name": name,
        "benchmark": "ZClawBench",
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "num_prompts": len(CHAT_PROMPTS),
    }

    # Extract spec decode metrics
    try:
        from vllm.v1.metrics.reader import Counter as VCounter, Vector
        raw_metrics = llm.get_metrics()
        num_drafts = 0
        num_accepted = 0
        acceptance_counts = [0, 0, 0]
        for m in raw_metrics:
            if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, VCounter):
                num_drafts += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(m, VCounter):
                num_accepted += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(m, Vector):
                for i in range(min(len(m.values), 3)):
                    acceptance_counts[i] += m.values[i]

        if num_drafts > 0:
            acceptance_length = 1 + (num_accepted / num_drafts)
            metrics["num_drafts"] = num_drafts
            metrics["num_accepted_tokens"] = num_accepted
            metrics["acceptance_length"] = round(acceptance_length, 3)
            for i, c in enumerate(acceptance_counts):
                metrics[f"acceptance_rate_pos_{i}"] = round(c / num_drafts, 4)
    except Exception as e:
        print(f"[WARN] Could not extract spec decode metrics: {e}")

    # Per-category throughput
    cat_tokens = {}
    cat_time = {}
    for i, out in enumerate(outputs):
        cat = PROMPT_CATEGORIES[i]
        toks = len(out.outputs[0].token_ids)
        cat_tokens[cat] = cat_tokens.get(cat, 0) + toks
    # Approximate per-category time by token ratio
    for cat in cat_tokens:
        frac = cat_tokens[cat] / total_tokens
        cat_time[cat] = round(elapsed * frac, 2)
    metrics["per_category_tokens"] = cat_tokens

    print(f"\n===== Results: {name} =====")
    for k, v in metrics.items():
        if k != "per_category_tokens":
            print(f"  {k}: {v}")

    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Results saved to {result_file}")

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return metrics


if __name__ == "__main__":
    evals_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["baseline", "aurora_spec", "exp14_ckpt54", "exp15_ckpt67"]

    all_results = []

    for eval_name in evals_to_run:
        if eval_name == "baseline":
            result = run_eval("baseline")
        elif eval_name == "aurora_spec":
            result = run_eval("aurora_spec", {
                "model": os.environ.get("AURORA_SPEC", "/data/models/Aurora-Spec-Minimax-M2.1"),
                "num_speculative_tokens": 3,
                "method": "eagle3",
            })
        elif eval_name == "exp14_ckpt54":
            result = run_eval("exp14_ckpt54", {
                "model": os.environ.get("EXP14_SPEC", "/data/output/minimax_m2.5_eagle3_aurora_arch/checkpoints/54"),
                "num_speculative_tokens": 3,
                "method": "eagle3",
            })
        elif eval_name == "exp15_ckpt67":
            result = run_eval("exp15_ckpt67", {
                "model": os.environ.get("EXP15_SPEC", "/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67"),
                "num_speculative_tokens": 3,
                "method": "eagle3",
            })
        else:
            print(f"[WARN] Unknown eval: {eval_name}")
            continue
        all_results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY — ZClawBench Eval")
    print(f"{'='*80}")
    baseline_tps = next((r["throughput_tokens_per_sec"] for r in all_results if r.get("name") == "baseline" and "error" not in r), 1)

    header = f"{'Model':<20} {'Tokens/s':>10} {'Speedup':>8} {'Acc Len':>8} {'Acc@0':>7} {'Acc@1':>7} {'Acc@2':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<20} {'ERROR':>10}   {r['error'][:50]}")
            continue
        tps = r["throughput_tokens_per_sec"]
        speedup = tps / baseline_tps if baseline_tps > 0 else 0
        acc_len = r.get("acceptance_length", "-")
        a0 = r.get("acceptance_rate_pos_0", "-")
        a1 = r.get("acceptance_rate_pos_1", "-")
        a2 = r.get("acceptance_rate_pos_2", "-")
        print(f"{r['name']:<20} {tps:>10.1f} {speedup:>7.2f}x {str(acc_len):>8} {str(a0):>7} {str(a1):>7} {str(a2):>7}")

    combined_file = os.path.join(OUTPUT_DIR, "combined_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Combined results saved to {combined_file}")
PYEOF

###############################################################################
# Run evaluations
###############################################################################
export BASE_MODEL OUTPUT_DIR TP MAX_MODEL_LEN
export AURORA_SPEC EXP14_SPEC EXP15_SPEC
export ZCLAWBENCH_PATH="/workspace/speculators/k8s/zclawbench_prompts.json"

for eval_name in baseline aurora_spec exp14_ckpt54 exp15_ckpt67; do
    echo ""
    echo ">>> Starting eval: $eval_name"
    rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${eval_name}" \
    python3 "$OUTPUT_DIR/eval_runner.py" "$eval_name" || echo "[WARN] $eval_name eval failed"
    echo ">>> Finished eval: $eval_name"
done

echo ""
echo "Done. Results in $OUTPUT_DIR/"
