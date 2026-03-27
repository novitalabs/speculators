#!/bin/bash
set -euo pipefail

if ! command -v python &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

cd /workspace/speculators
python3 k8s/patch_vllm_minimax_eagle3.py
find /usr/local/lib/python3.12/dist-packages/vllm/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_DIR="/data/output/minimax_m2.5_eval_exp16.5"
mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/eval_runner.py" << 'PYEOF'
import json, os, sys, time, gc

BASE_MODEL = "/data/models/MiniMax-M2.5"
OUTPUT_DIR = "/data/output/minimax_m2.5_eval_exp16.5"
TP = 4
MAX_MODEL_LEN = 8192

with open("/workspace/speculators/k8s/zclawbench_prompts.json") as f:
    CHAT_PROMPTS = [p["messages"] for p in json.load(f)]
print(f"[INFO] Loaded {len(CHAT_PROMPTS)} ZClawBench prompts")

def run_eval(name, spec_config=None):
    from vllm import LLM, SamplingParams
    result_file = os.path.join(OUTPUT_DIR, f"{name}_results.json")
    print(f"\n{'='*60}\n  Evaluating: {name}\n{'='*60}")
    llm_kwargs = {
        "model": BASE_MODEL, "tensor_parallel_size": TP,
        "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": 0.85,
        "trust_remote_code": True, "disable_log_stats": False, "seed": 42,
    }
    if spec_config:
        llm_kwargs["speculative_config"] = spec_config
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        r = {"name": name, "error": str(e)}
        with open(result_file, "w") as f: json.dump(r, f, indent=2)
        return r

    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512, ignore_eos=True)
    start = time.time()
    outputs = llm.chat(CHAT_PROMPTS, sp)
    elapsed = time.time() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    metrics = {"name": name, "total_tokens": total_tokens,
               "elapsed_seconds": round(elapsed, 2),
               "throughput_tokens_per_sec": round(total_tokens / elapsed, 2),
               "num_prompts": len(CHAT_PROMPTS)}
    try:
        from vllm.v1.metrics.reader import Counter as VC, Vector
        raw = llm.get_metrics(); nd = na = 0; ac = [0,0,0]
        for m in raw:
            if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, VC): nd += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(m, VC): na += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(m, Vector):
                for i in range(min(len(m.values),3)): ac[i] += m.values[i]
        if nd > 0:
            metrics["acceptance_length"] = round(1 + na/nd, 3)
            for i,c in enumerate(ac): metrics[f"acceptance_rate_pos_{i}"] = round(c/nd, 4)
    except Exception as e:
        print(f"[WARN] {e}")
    print(f"\n===== Results: {name} =====")
    for k,v in metrics.items(): print(f"  {k}: {v}")
    with open(result_file, "w") as f: json.dump(metrics, f, indent=2)
    del llm; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except: pass
    return metrics

if __name__ == "__main__":
    evals = sys.argv[1:] if len(sys.argv) > 1 else ["baseline", "aurora_spec_m21", "exp15_ckpt67", "exp16_ckpt53", "exp16_5_ckpt18"]
    specs = {
        "aurora_spec_m21": "/data/models/Aurora-Spec-Minimax-M2.1",
        "exp15_ckpt67": "/data/output/minimax_m2.5_eagle3_novita0320/checkpoints/67",
        "exp16_ckpt53": "/data/output/minimax_m2.5_eagle3_aurora_loss/checkpoints/53",
        "exp16_5_ckpt18": "/data/output/minimax_m2.5_eagle3_aurora_static/checkpoints/18",
    }
    all_results = []
    for name in evals:
        if name == "baseline":
            all_results.append(run_eval("baseline"))
        elif name in specs:
            all_results.append(run_eval(name, {"model": specs[name], "num_speculative_tokens": 3, "method": "eagle3"}))

    print(f"\n{'='*80}\n  FINAL SUMMARY\n{'='*80}")
    bl = next((r["throughput_tokens_per_sec"] for r in all_results if r.get("name")=="baseline" and "error" not in r), 1)
    hdr = f"{'Model':<22} {'Tok/s':>8} {'Speed':>7} {'AccLen':>7} {'Acc@0':>7} {'Acc@1':>7} {'Acc@2':>7}"
    print(hdr); print("-"*len(hdr))
    for r in all_results:
        if "error" in r: print(f"{r['name']:<22} {'ERR':>8}"); continue
        t=r["throughput_tokens_per_sec"]; s=t/bl if bl>0 else 0
        print(f"{r['name']:<22} {t:>8.1f} {s:>6.2f}x {str(r.get('acceptance_length','-')):>7} {str(r.get('acceptance_rate_pos_0','-')):>7} {str(r.get('acceptance_rate_pos_1','-')):>7} {str(r.get('acceptance_rate_pos_2','-')):>7}")
    with open(os.path.join(OUTPUT_DIR, "combined_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDone!")
PYEOF

for e in exp16_5_ckpt18; do
    echo ""
    echo ">>> Starting eval: $e"
    rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${e}" \
    python3 "$OUTPUT_DIR/eval_runner.py" "$e" || echo "[WARN] $e failed"
    echo ">>> Finished eval: $e"
done

echo ""
echo "All done. Results in $OUTPUT_DIR/"
