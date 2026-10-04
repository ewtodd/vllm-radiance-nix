"""Compare two saved next-token logits vectors (rel, corr, top-1/top-5 agreement)."""

import argparse
import os

import torch


def metrics(a, b):
    rel = ((a - b).norm() / b.norm().clamp_min(1e-9)).item()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    ta = a.topk(5).indices.tolist()
    tb = b.topk(5).indices.tolist()
    return rel, corr, ta[0] == tb[0], len(set(ta) & set(tb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="test logits .pt")
    parser.add_argument("--b", required=True, help="reference logits .pt")
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    a = torch.load(args.a, map_location="cpu", weights_only=False).float().view(-1)
    b = torch.load(args.b, map_location="cpu", weights_only=False).float().view(-1)
    rel, corr, top1, top5 = metrics(a, b)
    print("%s vs %s" % (os.path.basename(args.a), os.path.basename(args.b)))
    print("  rel=%.4f corr=%.4f top1=%s top5=%d/5" % (rel, corr, top1, top5))
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("  a top1=%r" % tokenizer.decode([a.argmax().item()]))
        print("  b top1=%r" % tokenizer.decode([b.argmax().item()]))
    return 0 if top1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
