"""Offline speed check for the A16 vs A8 paths: decode ms/step and prefill tok/s at 8k.

Same model, same flags except the one under test, so the ratio is like-for-like. Run once per
arm; the first generation is a warmup and is not measured.

    run_bench.sh /scratch/vllm-models/... decode  [ENV=...]
    run_bench.sh /scratch/vllm-models/... prefill [ENV=...]
"""

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["decode", "prefill", "both"], required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=16,          # the 27B hybrid model's mamba-cache budget at gpu_util 0.90
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=os.environ.get("BENCH_EAGER", "0") == "1",
    )
    tokenizer = llm.get_tokenizer()

    results = {}
    if args.mode in ("decode", "both"):
        prompt = "The capital of France is"
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
        start = time.perf_counter()
        outputs = llm.generate([prompt], SamplingParams(temperature=0.0,
                                                       max_tokens=args.decode_tokens))
        elapsed = time.perf_counter() - start
        produced = len(outputs[0].outputs[0].token_ids)
        results["decode"] = {
            "tokens": produced,
            "seconds": elapsed,
            "ms_per_step": 1000.0 * elapsed / produced,
            "tok_per_s": produced / elapsed,
        }

    if args.mode in ("prefill", "both"):
        ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
        while len(ids) < args.prefill_tokens:
            ids = ids + ids
        ids = ids[:args.prefill_tokens]
        llm.generate([{"prompt_token_ids": ids[:64]}],
                     SamplingParams(temperature=0.0, max_tokens=1))
        start = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0.0, max_tokens=1))
        elapsed = time.perf_counter() - start
        results["prefill"] = {
            "tokens": len(ids),
            "seconds": elapsed,
            "prefill_tok_per_s": len(ids) / elapsed,
        }

    result = results if args.mode == "both" else results[args.mode]
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
