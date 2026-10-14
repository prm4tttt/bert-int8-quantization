# Static INT8 Quantization of BERT-base (CPU)

Post-training **static** INT8 quantization of a BERT-base sentiment classifier (SST-2) using PyTorch's `fbgemm` x86 backend, plus a reproducible harness that measures accuracy, memory and latency.

## How it works

- **Static** PTQ quantizes both weights *and* activations to INT8. Activation scales and zero points are fixed ahead of time by running a small calibration set (512 SST-2 training sentences) through observers.
- Every `nn.Linear` in the encoder is wrapped as `QuantStub → Linear → DeQuantStub` ([`quantize.py`](quantize.py)). Those matmuls hold almost all of BERT's FLOPs, so they run as INT8 GEMMs via fbgemm. LayerNorm, GELU, softmax and embeddings stay in FP32.
- Weights use per-channel symmetric quantization; activations use per-tensor histogram observers (PyTorch's default `fbgemm` qconfig).
- The classifier head is left in FP32 (it's tiny and sensitive).

## Measurement harness

[`benchmark.py`](benchmark.py) is built to give stable numbers:
- fixed seeds (Python, NumPy, torch) and a fixed calibration subset
- fixed thread count (`--threads`)
- inputs padded to a fixed length so every trial does identical work
- warm-up runs before timing, then N trials → **median, p90 and std**
- resident memory sampled in a background thread during inference; the reported number is the
  increase over the pre-inference baseline (both models share one process, so absolute RSS would
  count both)
- weight memory measured as serialized `state_dict` size
- everything written to `results.json`

## Run

```bash
pip install -r requirements.txt
python benchmark.py --batch-sizes 1 8 32 --trials 50 --threads 4
```

Quick offline check (tiny random BERT, no downloads): `pytest -v test_quantize.py`.

The first run downloads the fine-tuned checkpoint and SST-2 (~450 MB). Needs an x86 CPU (fbgemm). On Apple Silicon, edit `quantize.py` to use the `qnnpack` backend.

> Note: PyTorch is moving quantization to `torchao`, so very recent versions print deprecation warnings for `torch.ao.quantization`. The code still runs. If a future release removes it, pin `torch<2.15`.

## Results

> Fill in from **your** runs (`results.json`). Include your CPU model.

CPU: _____  threads: _____

| Metric | FP32 | INT8 static | Change |
|---|---|---|---|
| SST-2 accuracy | | | |
| Weight memory (MB) | | | |
| Latency bs=1 (ms, median) | | | |
| Latency bs=8 (ms, median) | | | |
| Latency bs=32 (ms, median) | | | |
| Extra RSS during inference, bs=32 (MB) | | | |

## What I learned

_Fill in: where the speedup comes from, why the embeddings limit the memory savings, how the calibration set size affected accuracy._

## Next steps

- Quantize the embedding tables too (`float_qparams_weight_only_qconfig`) to cut memory further
- Compare against dynamic quantization (`torch.ao.quantization.quantize_dynamic`)
- Try FX graph-mode quantization or `torchao`
- Sweep calibration set size (32 / 128 / 512 / 2048)
