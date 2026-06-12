# PCDformer real second innovation: RevIN + Trend-Seasonal Residual Forecast Adapter

This package is based on the previously saved ACR/HCDA code. The broken AMTF/LPD full-open version is not used.

## What changed

A real trainable forecasting module is added instead of dataset-specific if-else switching:

1. Reversible Instance Normalization (`--use_revin 1`)
   - normalizes each sample and variable before PCDformer,
   - denormalizes final prediction back to the original scale.

2. Trend-Seasonal Residual Forecast Adapter (`--use_residual_adapter 1`)
   - decomposes the original input into trend and seasonal parts,
   - learns a conservative DLinear-style residual prediction,
   - adds it to the PCDformer output with a learnable gate.

The adapter is initialized conservatively with `--adapter_gate_init -4.0`, so it starts as a small correction branch rather than a disruptive replacement.

## Suggested command suffix

Use this suffix on top of the ACR command:

```bash
--use_revin 1 --use_residual_adapter 1 --adapter_gate_init -4.0
```

If the residual adapter is too weak, try:

```bash
--adapter_gate_init -3.0
```

## Important

This version does not add AMTF or LPD. Those modules hurt the current results in your logs.
