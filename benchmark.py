"""Reproducible CPU benchmark: FP32 BERT-base vs. static INT8 BERT-base.

Measures, for each model:
  - accuracy on the SST-2 validation set (872 sentences)
  - weight memory (serialized state_dict size)
  - extra process memory (RSS above the pre-inference baseline) during
    inference, per batch size. Both models live in the same process, so the
    absolute peak RSS includes both; the delta is what the forward pass costs.
  - latency per batch: warm-up runs, then N timed trials -> median / p90 / std

Run:
  python benchmark.py                       # defaults
  python benchmark.py --batch-sizes 1 8 32 --trials 50 --threads 4
"""

import argparse
import json
import os
import platform
import random
import statistics
import threading
import time

import numpy as np
import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from quantize import model_size_mb, static_quantize

# A BERT-base checkpoint already fine-tuned on SST-2.
DEFAULT_MODEL = "textattack/bert-base-uncased-SST-2"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PeakRSS:
    """Samples this process's resident memory in a background thread."""

    def __init__(self, interval=0.002):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self):
        return self.peak / 2**20

    @property
    def extra_mb(self):
        return (self.peak - self.baseline) / 2**20


def make_batches(tokenizer, sentences, labels, batch_size, max_len):
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], padding="max_length",
                        truncation=True, max_length=max_len, return_tensors="pt")
        yield dict(enc), torch.tensor(labels[i:i + batch_size])


@torch.no_grad()
def accuracy(model, tokenizer, sentences, labels, max_len):
    correct = 0
    for batch, y in make_batches(tokenizer, sentences, labels, 32, max_len):
        correct += (model(**batch).logits.argmax(-1) == y).sum().item()
    return correct / len(labels)


@torch.no_grad()
def latency(model, batch, warmup, trials):
    for _ in range(warmup):
        model(**batch)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        model(**batch)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p90_ms": times[int(0.9 * (len(times) - 1))],
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    print(f"CPU: {platform.processor() or platform.machine()} | torch {torch.__version__} | "
          f"threads={torch.get_num_threads()} | engine=fbgemm")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32 = AutoModelForSequenceClassification.from_pretrained(args.model).eval()

    sst2 = load_dataset("nyu-mll/glue", "sst2")
    # list(...) because newer `datasets` versions return lazy Column objects.
    val_s, val_y = list(sst2["validation"]["sentence"]), list(sst2["validation"]["label"])
    train = sst2["train"].shuffle(seed=args.seed).select(range(args.calib_samples))
    calib = [b for b, _ in make_batches(tokenizer, list(train["sentence"]), list(train["label"]),
                                        32, args.max_len)]

    int8 = static_quantize(fp32, calib, backend="fbgemm")

    results = {"config": vars(args), "models": {}}
    for name, model in [("fp32", fp32), ("int8_static", int8)]:
        print(f"\n== {name} ==")
        r = {"weights_mb": model_size_mb(model)}
        r["accuracy"] = accuracy(model, tokenizer, val_s, val_y, args.max_len)
        print(f"weights {r['weights_mb']:.1f} MB | SST-2 acc {r['accuracy']:.4f}")
        r["per_batch"] = {}
        for bs in args.batch_sizes:
            batch, _ = next(make_batches(tokenizer, val_s[:bs], val_y[:bs], bs, args.max_len))
            with PeakRSS() as mem:
                lat = latency(model, batch, args.warmup, args.trials)
            lat["peak_rss_mb"] = mem.peak_mb
            lat["extra_rss_mb"] = mem.extra_mb
            r["per_batch"][bs] = lat
            print(f"  bs={bs:<3} median {lat['median_ms']:8.2f} ms  p90 {lat['p90_ms']:8.2f}  "
                  f"std {lat['std_ms']:6.2f}  extra RSS {lat['extra_rss_mb']:.0f} MB")
        results["models"][name] = r

    f, q = results["models"]["fp32"], results["models"]["int8_static"]
    print("\n== Summary (INT8 vs FP32) ==")
    print(f"weight memory: -{100 * (1 - q['weights_mb'] / f['weights_mb']):.1f}%")
    print(f"accuracy drop: {100 * (f['accuracy'] - q['accuracy']):.2f} points")
    for bs in args.batch_sizes:
        d = 100 * (1 - q["per_batch"][bs]["median_ms"] / f["per_batch"][bs]["median_ms"])
        print(f"latency bs={bs}: -{d:.1f}%")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
