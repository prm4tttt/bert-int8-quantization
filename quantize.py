"""Static INT8 post-training quantization for BERT-base on CPU (fbgemm).

Static PTQ = weights AND activations are INT8, with activation scales/zero
points picked ahead of time by running a calibration set through observers.
(Compare with dynamic quantization, where activation scales are computed on
the fly at every forward pass.)

Approach: every nn.Linear inside the encoder is wrapped as
    QuantStub -> Linear -> DeQuantStub
so the matmuls (where ~all of BERT's FLOPs and parameters live) run in INT8,
while LayerNorm, GELU, softmax and embeddings stay in FP32.
"""

import copy

import torch
import torch.nn as nn
from torch.ao.quantization import (DeQuantStub, QuantStub, convert,
                                   get_default_qconfig, prepare)


class QuantizableLinear(nn.Module):
    """Wraps a float Linear with quant/dequant boundaries."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quant = QuantStub()
        self.linear = linear
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.linear(self.quant(x)))


def _wrap_linears(module: nn.Module, skip: tuple[str, ...]) -> int:
    count = 0
    for name, child in module.named_children():
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizableLinear(child))
            count += 1
        else:
            count += _wrap_linears(child, skip)
    return count


@torch.no_grad()
def static_quantize(model: nn.Module, calibration_batches, backend: str = "fbgemm",
                    skip_modules: tuple[str, ...] = ("classifier",)) -> nn.Module:
    """Returns a statically quantized copy of `model`.

    calibration_batches: iterable of dicts that can be passed as model(**batch).
    skip_modules: child names left in FP32 (the classifier head is tiny and
                  quantizing it tends to cost accuracy for no speed gain).
    """
    torch.backends.quantized.engine = backend
    qmodel = copy.deepcopy(model).eval()

    n = _wrap_linears(qmodel, skip_modules)

    # Only the wrapped linears get a qconfig; everything else stays float.
    qconfig = get_default_qconfig(backend)
    qmodel.qconfig = None
    for m in qmodel.modules():
        if isinstance(m, QuantizableLinear):
            m.qconfig = qconfig

    prepare(qmodel, inplace=True)          # insert observers
    for batch in calibration_batches:      # observers record activation ranges
        qmodel(**batch)
    convert(qmodel, inplace=True)          # swap to INT8 kernels
    print(f"Quantized {n} Linear layers to INT8 ({backend}).")
    return qmodel


def model_size_mb(model: nn.Module) -> float:
    """Serialized state_dict size: a fair measure of weight memory."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20
