"""Offline next-token logits for a checkpoint, captured from the vLLM sampler.

Usage:
    RADIANCE_LOGITS_OUT=/scratch/w16/logits_x.pt \
    PYTHONPATH=/scratch/w16/shadow:/scratch/w16 python logits_serve.py \
        --model /scratch/vllm-models/Swift-1.5-...-Quark-RTN-MXFP4-W4A16 \
        --prompt "The capital of France is" --attention-backend R4D

Runs the exact model/kernel stack under test (env flags are read by the radiance modules),
generates one greedy token and saves the full vocab logits vector. compare_logits.py then
relates it to the saved bf16-dequantized reference vectors.
"""

import argparse

import torch
from vllm import LLM, SamplingParams

from logits_capture import LogitsCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--quantization", default=None)
    args = parser.parse_args()

    kwargs = {"logits_processors": [LogitsCapture]}
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.quantization:
        kwargs["quantization"] = args.quantization

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=True,
        **kwargs,
    )
    outputs = llm.generate([args.prompt], SamplingParams(temperature=0.0, max_tokens=1))
    tokenizer = llm.get_tokenizer()
    print("generated:", repr(outputs[0].outputs[0].text))
    saved = torch.load(__import__("os").environ["RADIANCE_LOGITS_OUT"],
                       map_location="cpu", weights_only=False)
    top = saved.topk(5)
    print("saved logits", tuple(saved.shape), "absmean", saved.abs().mean().item())
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        print("   %8.3f %r" % (value, tokenizer.decode([index])))


if __name__ == "__main__":
    main()
