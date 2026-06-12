# PCDformer on the Autoformer Codebase

This project adds a paper-aligned PCDformer implementation to the original Autoformer repository.

## Changed / added files

- `models/PCDformer.py`: complete PCDformer model, aligned with the paper's module equations and Fig. 4 M-inception design.
- `exp/exp_main.py`: registers `PCDformer` in `model_dict`.
- `run.py`: adds the `--top_k` argument and includes PCDformer in the model options.
- `scripts/PCDformer_script/`: runnable example scripts for ETTh1, ETTm1, and Weather.

## Paper-aligned implementation notes

Autoformer originally uses time-step tokens: `[B, L, N] -> [B, L, d_model]`.
PCDformer uses variable tokens: `[B, L, N] -> [B, N, d_model]`.

Implemented modules:

1. Parallel processing layer:
   - transposes `[B, L, N]` into variable tokens `[B, N, L]`;
   - applies the paper's Dim-trans operation to form `[B, N, P, P]`, zero-padding when `sqrt(L)` is not an integer;
   - applies independent M-inception processing for each variable;
   - reshapes back to `[B, N, L]` and projects to `[B, N, d_model]`.
2. M-inception:
   - follows Fig. 4: four branches with `AvgPool -> 1x1 Conv(24,0)`, `1x1 Conv(24,0)`, `1x1 Conv(16,0) -> 5x5 Conv(24,2)`, and `1x1 Conv(16,0) -> 3x3 Conv(24,1) -> 3x3 Conv(24,1)`;
   - sums the branch outputs, applies ReLU, then averages over the 24 feature maps.
   - uses grouped convolutions so that each variable has independent parameters while remaining efficient.
3. Series decomposition:
   - uses `AvePool(Padding(X_series))` for the trend and subtracts it from the input to get the seasonal component.
4. Sparse self-attention:
   - applies Top-k masking over variable-token attention scores;
   - divides scores by `sqrt(d_model)`, matching the paper equations.
5. Encoder, decoder, and trend module:
   - encoder has sparse self-attention, residual addition, decomposition, normalization, feed-forward, and decomposition;
   - decoder has sparse self-attention, dense cross-attention, feed-forward, and three trend residuals;
   - trend module uses the trend component of `SD(X_enc)` plus `Mean(X_enc)`, followed by a linear projection.
6. Final projection:
   - `[B, N, d_model] -> [B, pred_len, N]`.

## Example command

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id PCDformer_ETTh1_201_24 \
  --model PCDformer \
  --data ETTh1 \
  --features M \
  --seq_len 201 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 1024 \
  --n_heads 8 \
  --moving_avg 25 \
  --top_k 1 \
  --dropout 0.05 \
  --batch_size 16 \
  --learning_rate 0.00008 \
  --train_epochs 10 \
  --patience 3 \
  --des PCDformer \
  --itr 1
```

`label_len` is kept only to satisfy the original Autoformer data pipeline. PCDformer internally builds the decoder input as `[observed history, zeros for future]`, so `label_len=0` is recommended.

## Reproducibility note

The paper does not provide every low-level engineering choice, such as exact parameter initialization, seed handling, optimizer scheduler internals, and the authors' full training scripts. This code therefore targets architectural and algorithmic consistency with the paper. Exact numerical reproduction of every table entry may still vary by GPU, PyTorch version, random seed, and dataset preprocessing details.
