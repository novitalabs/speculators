#!/bin/bash
set -euo pipefail

###############################################################################
# MiniMax-M2.5 Eagle3 Evaluation - Compare draft models
# Tests: baseline (no spec), Aurora-Spec, our Novita-trained model
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

BASE_MODEL="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
AURORA_SPEC="/data/models/Aurora-Spec-Minimax-M2.1"
NOVITA_SPEC="/data/output/minimax_m2.5_eagle3_novita_vllm"
NOVITA2_SPEC="/data/output/minimax_m2.5_eagle3_novita2_vllm"
OUTPUT_DIR="/data/output/minimax_m2.5_eval6"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

mkdir -p "$OUTPUT_DIR"

cd /workspace/speculators

###############################################################################
# Patch vLLM: add Eagle3 support for MiniMax-M2 models
###############################################################################
python3 /workspace/speculators/k8s/patch_vllm_minimax_eagle3.py
# Clear ALL bytecode caches to ensure patched .py files are used by subprocesses
find /usr/local/lib/python3.12/dist-packages/vllm/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
# Prevent Python from writing new .pyc files (ensures spawned workers read patched .py)
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
import random

BASE_MODEL = os.environ["BASE_MODEL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
TP = int(os.environ["TP"])
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"])
NOVITA_DATA = os.environ.get("NOVITA_DATA", "/data/datasets/novita20260309/conversations.jsonl")

# Load Novita prompts: sample 10 conversations, use system+user messages as chat context
def load_novita_prompts(path, n=10, seed=42):
    """Load multi-turn conversations from Novita data as chat prompts."""
    convs = []
    with open(path) as f:
        for line in f:
            convs.append(json.loads(line))
    random.seed(seed)
    sampled = random.sample(convs, min(n, len(convs)))
    prompts = []
    for c in sampled:
        msgs = c["conversations"]
        # Take system + first user message (truncate user content to fit in context)
        chat = []
        for m in msgs:
            if m["role"] == "system":
                chat.append({"role": "system", "content": m["content"][:2000]})
            elif m["role"] == "user":
                chat.append({"role": "user", "content": m["content"][:2000]})
                break  # only first user turn
        if any(m["role"] == "user" for m in chat):
            prompts.append(chat)
    return prompts

CHAT_PROMPTS = load_novita_prompts(NOVITA_DATA)
print(f"[INFO] Loaded {len(CHAT_PROMPTS)} Novita chat prompts")


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
        error_result = {"name": name, "error": str(e)}
        with open(result_file, "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result

    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=512, ignore_eos=True,
    )

    print(f"[INFO] Running chat inference on {len(CHAT_PROMPTS)} Novita prompts (512 tokens each)...")
    start = time.time()
    outputs = llm.chat(CHAT_PROMPTS, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    metrics = {
        "name": name,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "num_prompts": len(CHAT_PROMPTS),
    }

    # Extract spec decode metrics
    try:
        from vllm.v1.metrics.reader import Counter, Vector
        raw_metrics = llm.get_metrics()
        num_drafts = 0
        num_accepted = 0
        acceptance_counts = [0, 0, 0]
        for m in raw_metrics:
            if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, Counter):
                num_drafts += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(m, Counter):
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

    print(f"\n===== Results: {name} =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Results saved to {result_file}")

    # Clean up GPU memory
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return metrics


if __name__ == "__main__":
    # Get which evals to run from command line args
    evals_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["baseline", "aurora_spec", "novita2_spec"]

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
        elif eval_name == "novita_spec":
            result = run_eval("novita_spec", {
                "model": os.environ.get("NOVITA_SPEC", "/data/output/minimax_m2.5_eagle3_novita/checkpoints/9"),
                "num_speculative_tokens": 3,
                "method": "eagle3",
            }, enforce_eager=True)
        elif eval_name == "novita2_spec":
            result = run_eval("novita2_spec", {
                "model": os.environ.get("NOVITA2_SPEC", "/data/output/minimax_m2.5_eagle3_novita2_vllm"),
                "num_speculative_tokens": 3,
                "method": "eagle3",
            })
        else:
            print(f"[WARN] Unknown eval: {eval_name}")
            continue
        all_results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
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

    # Save combined results
    combined_file = os.path.join(OUTPUT_DIR, "combined_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Combined results saved to {combined_file}")
PYEOF

###############################################################################
# Run evaluations
###############################################################################
export BASE_MODEL OUTPUT_DIR TP MAX_MODEL_LEN
export AURORA_SPEC NOVITA_SPEC NOVITA2_SPEC

# Run each eval in a separate process for clean GPU memory
# Use per-eval inductor cache dirs to avoid torch.compile cache conflicts
# (different draft models have different weight shapes)
for eval_name in baseline novita2_spec aurora_spec; do
    echo ""
    echo ">>> Starting eval: $eval_name"
    # Clear vLLM compile cache to avoid shape conflicts between different draft models
    rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${eval_name}" \
    python3 "$OUTPUT_DIR/eval_runner.py" "$eval_name" || echo "[WARN] $eval_name eval failed"
    echo ">>> Finished eval: $eval_name"
done

# Print combined summary
python3 -c "
import json, os, glob
results = []
for f in sorted(glob.glob(os.path.join('$OUTPUT_DIR', '*_results.json'))):
    if 'combined' not in f:
        results.append(json.load(open(f)))
baseline_tps = next((r['throughput_tokens_per_sec'] for r in results if r.get('name')=='baseline' and 'error' not in r), 1)
print()
print('='*80)
print('  FINAL SUMMARY')
print('='*80)
header = f\"{'Model':<20} {'Tokens/s':>10} {'Speedup':>8} {'Acc Len':>8} {'Acc@0':>7} {'Acc@1':>7} {'Acc@2':>7}\"
print(header)
print('-' * len(header))
for r in results:
    if 'error' in r:
        print(f\"{r['name']:<20} {'ERROR':>10}   {r.get('error','')[:50]}\")
        continue
    tps = r['throughput_tokens_per_sec']
    speedup = tps / baseline_tps if baseline_tps > 0 else 0
    acc_len = r.get('acceptance_length', '-')
    a0 = r.get('acceptance_rate_pos_0', '-')
    a1 = r.get('acceptance_rate_pos_1', '-')
    a2 = r.get('acceptance_rate_pos_2', '-')
    print(f\"{r['name']:<20} {tps:>10.1f} {speedup:>7.2f}x {str(acc_len):>8} {str(a0):>7} {str(a1):>7} {str(a2):>7}\")
with open(os.path.join('$OUTPUT_DIR', 'combined_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n[INFO] Combined results saved to $OUTPUT_DIR/combined_results.json')
"

echo ""
echo "Done. Results in $OUTPUT_DIR/"
