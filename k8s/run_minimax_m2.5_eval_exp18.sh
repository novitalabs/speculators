#!/bin/bash
set -euo pipefail

###############################################################################
# Exp18 Eval: Novita (10 prompts) + ZClawBench (116 prompts)
# Compare exp18_ckpt42 against baseline
###############################################################################

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

BASE_MODEL="${MODEL_PATH:-/data/models/MiniMax-M2.5}"
EXP18_CKPT="${EXP18_CKPT:-/data/output/minimax_m2.5_eagle3_exp18/checkpoints/42}"
OUTPUT_DIR="/data/output/minimax_m2.5_eval_exp18"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

mkdir -p "$OUTPUT_DIR"
cd /workspace/speculators

# Patch vLLM
python3 /workspace/speculators/k8s/patch_vllm_minimax_eagle3.py
find /usr/local/lib/python3.12/dist-packages/vllm/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

###############################################################################
cat > "$OUTPUT_DIR/eval_runner.py" << 'PYEOF'
import json, os, sys, time, gc, random

BASE_MODEL = os.environ["BASE_MODEL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
TP = int(os.environ["TP"])
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"])

def load_novita_prompts(path, n=10, seed=42):
    convs = []
    with open(path) as f:
        for line in f:
            convs.append(json.loads(line))
    random.seed(seed)
    sampled = random.sample(convs, min(n, len(convs)))
    prompts = []
    for c in sampled:
        msgs = c["conversations"]
        chat = []
        for m in msgs:
            if m["role"] == "system":
                chat.append({"role": "system", "content": m["content"][:2000]})
            elif m["role"] == "user":
                chat.append({"role": "user", "content": m["content"][:2000]})
                break
        if any(m["role"] == "user" for m in chat):
            prompts.append(chat)
    return prompts

def load_zclawbench_prompts(path):
    with open(path) as f:
        raw = json.load(f)
    return [item["messages"] for item in raw]

def run_eval(name, prompts, benchmark, spec_config=None):
    from vllm import LLM, SamplingParams
    result_file = os.path.join(OUTPUT_DIR, f"{name}_{benchmark}_results.json")
    print(f"\n{'='*60}\n  Evaluating: {name} ({benchmark}, {len(prompts)} prompts)\n{'='*60}")

    llm_kwargs = {
        "model": BASE_MODEL, "tensor_parallel_size": TP,
        "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": 0.95,
        "trust_remote_code": True, "disable_log_stats": False, "seed": 42,
    }
    if spec_config:
        llm_kwargs["speculative_config"] = spec_config

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback; traceback.print_exc()
        r = {"name": name, "benchmark": benchmark, "error": str(e)}
        with open(result_file, "w") as f: json.dump(r, f, indent=2)
        return r

    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512, ignore_eos=True)
    start = time.time()
    outputs = llm.chat(prompts, sp)
    elapsed = time.time() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    metrics = {
        "name": name, "benchmark": benchmark,
        "total_tokens": total_tokens, "elapsed_seconds": round(elapsed, 2),
        "throughput_tokens_per_sec": round(total_tokens / elapsed, 2),
        "num_prompts": len(prompts),
    }

    try:
        from vllm.v1.metrics.reader import Counter, Vector
        raw_m = llm.get_metrics()
        nd, na, ac = 0, 0, [0, 0, 0]
        for m in raw_m:
            if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, Counter): nd += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(m, Counter): na += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(m, Vector):
                for i in range(min(len(m.values), 3)): ac[i] += m.values[i]
        if nd > 0:
            metrics["acceptance_length"] = round(1 + na / nd, 3)
            metrics["num_drafts"] = nd
            metrics["num_accepted_tokens"] = na
            for i, c in enumerate(ac):
                metrics[f"acceptance_rate_pos_{i}"] = round(c / nd, 4)
    except Exception as e:
        print(f"[WARN] Spec metrics: {e}")

    print(f"\n===== Results: {name} ({benchmark}) =====")
    for k, v in metrics.items(): print(f"  {k}: {v}")
    with open(result_file, "w") as f: json.dump(metrics, f, indent=2)
    del llm; gc.collect()
    try: import torch; torch.cuda.empty_cache()
    except: pass
    return metrics

if __name__ == "__main__":
    novita_prompts = load_novita_prompts(
        os.environ.get("NOVITA_DATA", "/data/datasets/novita20260312_eval/conversations.jsonl"))
    zclaw_prompts = load_zclawbench_prompts("/workspace/speculators/k8s/zclawbench_prompts.json")

    ckpt_path = os.environ.get("EXP18_CKPT", "")
    spec_config = {"model": ckpt_path, "num_speculative_tokens": 3, "method": "eagle3"}

    all_results = []

    # Novita eval
    r = run_eval("exp18_ckpt42", novita_prompts, "novita", spec_config)
    all_results.append(r)

    # ZClawBench eval
    r = run_eval("exp18_ckpt42", zclaw_prompts, "zclawbench", spec_config)
    all_results.append(r)

    # Summary
    print(f"\n{'='*80}\n  SUMMARY — Exp18 ckpt42 Eval\n{'='*80}")
    header = f"{'Benchmark':<15} {'Tokens/s':>10} {'Acc@0':>7} {'Acc@1':>7} {'Acc@2':>7} {'Acc Len':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{r['benchmark']:<15} ERROR: {r['error'][:50]}")
            continue
        print(f"{r['benchmark']:<15} {r['throughput_tokens_per_sec']:>10.1f} "
              f"{r.get('acceptance_rate_pos_0', '-'):>7} "
              f"{r.get('acceptance_rate_pos_1', '-'):>7} "
              f"{r.get('acceptance_rate_pos_2', '-'):>7} "
              f"{str(r.get('acceptance_length', '-')):>8}")

    with open(os.path.join(OUTPUT_DIR, "combined_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Results saved to {OUTPUT_DIR}/combined_results.json")
PYEOF

###############################################################################
export BASE_MODEL OUTPUT_DIR TP MAX_MODEL_LEN EXP18_CKPT
export NOVITA_DATA="${NOVITA_DATA:-/data/datasets/novita20260312_eval/conversations.jsonl}"

rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
python3 "$OUTPUT_DIR/eval_runner.py"

echo "Done. Results in $OUTPUT_DIR/"
