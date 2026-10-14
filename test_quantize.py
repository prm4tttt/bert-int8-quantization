"""Offline tests for static_quantize on a tiny randomly-initialised BERT (no downloads).

  pytest -v test_quantize.py
"""
import torch
from torch.ao.nn.quantized import Linear as QLinear
from transformers import BertConfig, BertForSequenceClassification

from quantize import model_size_mb, static_quantize


def tiny_bert():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=200, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=128, max_position_embeddings=64, num_labels=2)
    return BertForSequenceClassification(cfg).eval()


def batches(n=8, bs=4, seq=32):
    g = torch.Generator().manual_seed(1)
    for _ in range(n):
        yield {"input_ids": torch.randint(0, 200, (bs, seq), generator=g),
               "attention_mask": torch.ones(bs, seq, dtype=torch.long)}


def test_encoder_linears_become_int8_and_classifier_stays_fp32():
    q = static_quantize(tiny_bert(), list(batches()))
    n_int8 = sum(isinstance(m, QLinear) for m in q.modules())
    assert n_int8 == 2 * 6 + 1          # per layer: Q, K, V, attn-out, FFN-in, FFN-out; plus pooler
    assert isinstance(q.classifier, torch.nn.Linear) and not isinstance(q.classifier, QLinear)


def test_quantized_outputs_track_fp32():
    fp32 = tiny_bert()
    q = static_quantize(fp32, list(batches()))
    batch = next(batches(n=1, bs=16))
    with torch.no_grad():
        a, b = fp32(**batch).logits, q(**batch).logits
    assert torch.isfinite(b).all()
    assert torch.allclose(a, b, atol=0.1), (a - b).abs().max()


def test_weights_shrink():
    fp32 = tiny_bert()
    q = static_quantize(fp32, list(batches()))
    assert model_size_mb(q) < model_size_mb(fp32)
