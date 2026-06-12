# PCDformer ACR + HCDA Second Innovation

This version keeps the first innovation, Adaptive Core-Routed Sparse Attention (ACR), and adds a small second innovation named Hybrid Core-Direct Attention Rescue (HCDA).

## Motivation

The pure ACR branch is efficient and strong on high-dimensional datasets such as Traffic, but on small-variable datasets such as ETTh1, ETTm1 and Weather it can over-compress direct variable interactions through a small set of core tokens. HCDA keeps pure ACR for high-dimensional datasets and activates a direct top-k sparse rescue branch only when the variable count is small or medium.

## What changed

Main edited files:

- `models/PCDformer.py`
- `run.py`

New argument:

```bash
--hybrid_direct_threshold 64
```

Default behavior:

- If `enc_in <= hybrid_direct_threshold`, ACR is fused with a direct top-k sparse attention branch.
- If `enc_in > hybrid_direct_threshold`, the model keeps pure ACR to preserve the Traffic advantage and avoid the expensive full N x N direct branch.
- Set `--hybrid_direct_threshold 0` to disable the second innovation and recover the first-innovation ACR behavior.

## Recommended usage

For ETTh1, ETTm1 and Weather, keep the default:

```bash
--attention_type acr --core_num 0 --route_activation sparsemax --hybrid_direct_threshold 64
```

For Traffic and Electricity, the default also works. Traffic has `enc_in=862`, so the direct rescue branch is automatically disabled and the first ACR advantage is preserved.

## Why this is conservative

The model backbone, data loader, training loop, PCDformer encoder-decoder structure, trend module and projection head are unchanged. Only the ACR attention module is extended with an optional direct sparse branch. This makes the second innovation easy to ablate and less likely to break already-good results.
