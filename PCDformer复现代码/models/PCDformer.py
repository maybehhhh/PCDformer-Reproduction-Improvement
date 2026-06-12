import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgLastDim(nn.Module):
    """
    Moving average along the last dimension.

    In PCDformer, each variable is treated as a token and the token feature
    dimension is d_model, so the series decomposition module operates on the
    last dimension of a tensor shaped [B, N, D].
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvgLastDim, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, N, D]
        b, n, d = x.shape
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, pad_len)
        end = x[:, :, -1:].repeat(1, 1, pad_len)
        x_pad = torch.cat([front, x, end], dim=-1)
        x_pad = x_pad.reshape(b * n, 1, d + 2 * pad_len)
        trend = self.avg(x_pad).reshape(b, n, d)
        return trend


class SeriesDecompLastDim(nn.Module):
    """Series decomposition: X_t = AvePool(Padding(X)), X_s = X - X_t."""
    def __init__(self, kernel_size):
        super(SeriesDecompLastDim, self).__init__()
        self.moving_avg = MovingAvgLastDim(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class PaperMInception(nn.Module):
    """
    M-inception block following Fig. 4 of the PCDformer paper.

    The paper applies n parallel M-inception layers, one for each variable.
    This implementation is mathematically equivalent but vectorized with
    grouped convolutions: each variable group owns independent parameters.

    Branches:
      1) AvgPool -> 1x1 Conv, 24 kernels, padding 0
      2) 1x1 Conv, 24 kernels, padding 0
      3) 1x1 Conv, 16 kernels -> 5x5 Conv, 24 kernels, padding 2
      4) 1x1 Conv, 16 kernels -> 3x3 Conv, 24 kernels, padding 1
                         -> 3x3 Conv, 24 kernels, padding 1
    The branch outputs are summed, passed through ReLU, and averaged across
    the 24 feature maps to recover one output image per variable.

    Input/Output: [B, N, P, P]
    """
    def __init__(self, n_vars, branch_channels=24, reduce_channels=16, dropout=0.0):
        super(PaperMInception, self).__init__()
        self.n_vars = n_vars
        self.branch_channels = branch_channels
        self.reduce_channels = reduce_channels

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # grouped conv with out_channels = n_vars * channels is equivalent to
        # applying one independent conv module to each variable image.
        self.pool_conv = nn.Conv2d(n_vars, n_vars * branch_channels,
                                   kernel_size=1, padding=0, groups=n_vars, bias=True)
        self.conv1 = nn.Conv2d(n_vars, n_vars * branch_channels,
                               kernel_size=1, padding=0, groups=n_vars, bias=True)

        self.conv3_reduce = nn.Conv2d(n_vars, n_vars * reduce_channels,
                                      kernel_size=1, padding=0, groups=n_vars, bias=True)
        self.conv3 = nn.Conv2d(n_vars * reduce_channels, n_vars * branch_channels,
                               kernel_size=5, padding=2, groups=n_vars, bias=True)

        self.conv5_reduce = nn.Conv2d(n_vars, n_vars * reduce_channels,
                                      kernel_size=1, padding=0, groups=n_vars, bias=True)
        self.conv5_a = nn.Conv2d(n_vars * reduce_channels, n_vars * branch_channels,
                                 kernel_size=3, padding=1, groups=n_vars, bias=True)
        self.conv5_b = nn.Conv2d(n_vars * branch_channels, n_vars * branch_channels,
                                 kernel_size=3, padding=1, groups=n_vars, bias=True)

        self.dropout = nn.Dropout2d(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_feature_maps(self, x):
        # [B, N*C, P, P] -> [B, N, C, P, P]
        b, _, p1, p2 = x.shape
        return x.view(b, self.n_vars, self.branch_channels, p1, p2)

    def forward(self, x):
        # x: [B, N, P, P]
        branch_pool = self.pool_conv(self.avg_pool(x))
        branch_1x1 = self.conv1(x)
        branch_5x5 = self.conv3(self.conv3_reduce(x))
        branch_3x3_stack = self.conv5_b(self.conv5_a(self.conv5_reduce(x)))

        fused = branch_pool + branch_1x1 + branch_5x5 + branch_3x3_stack
        fused = F.relu(fused)
        fused = self.dropout(fused)
        fused = self._split_feature_maps(fused).mean(dim=2)
        return fused


class ParallelProcessingLayer(nn.Module):
    """
    Paper-aligned PCDformer parallel processing layer.

    It implements the pipeline:
      X_enc [B, L, N] -> variable tokens [B, N, L]
      -> Dim-trans [B, N, P, P] with zero padding when needed
      -> n independent M-inception layers, vectorized as grouped convs
      -> reshape [B, N, L]
      -> Linear [B, N, d_model]
    """
    def __init__(self, seq_len, n_vars, d_model, dropout=0.0):
        super(ParallelProcessingLayer, self).__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.p = int(math.ceil(math.sqrt(seq_len)))
        self.pad_len = self.p * self.p - seq_len
        self.conv = PaperMInception(n_vars=n_vars, dropout=dropout)
        self.linear = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1).contiguous()
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode='constant', value=0.0)
        # [B, N, P, P]
        x = x.reshape(x.shape[0], self.n_vars, self.p, self.p)
        x = self.conv(x)
        # [B, N, P*P] -> crop padded positions -> [B, N, L]
        x = x.reshape(x.shape[0], self.n_vars, self.p * self.p)[:, :, :self.seq_len]
        x = self.linear(x)
        return self.dropout(x)


class VariableEmbedding(nn.Module):
    """Linear embedding for decoder variable tokens: [B, N, L + tau] -> [B, N, D]."""
    def __init__(self, input_len, d_model, dropout=0.0):
        super(VariableEmbedding, self).__init__()
        self.proj = nn.Linear(input_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(x))


def apply_attention_mask(scores, attn_mask):
    if attn_mask is None:
        return scores
    mask = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
    return scores.masked_fill(mask, float('-inf'))


class PaperSparseMultiHeadAttention(nn.Module):
    """
    Top-k sparse multi-head self-attention over variable tokens.

    The score scaling follows the paper formulas (3)-(5), i.e. division by
    sqrt(d_model), rather than PyTorch's default per-head sqrt(d_head).
    """
    def __init__(self, d_model, n_heads, top_k=1, dropout=0.1, output_attention=False):
        super(PaperSparseMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.output_attention = output_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        b, l_q, _ = queries.shape
        _, l_k, _ = keys.shape

        q = self.q_proj(queries).view(b, l_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(keys).view(b, l_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(values).view(b, l_k, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = apply_attention_mask(scores, attn_mask)

        k_keep = min(max(int(self.top_k), 1), l_k)
        if k_keep < l_k:
            top_values, top_indices = torch.topk(scores, k=k_keep, dim=-1)
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, top_indices, top_values)
            scores = sparse_scores

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, l_q, self.d_model)
        out = self.out_proj(out)
        return out, attn if self.output_attention else None


class PaperDenseMultiHeadAttention(nn.Module):
    """Dense multi-head attention with the same sqrt(d_model) score scale."""
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super(PaperDenseMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.output_attention = output_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        b, l_q, _ = queries.shape
        _, l_k, _ = keys.shape

        q = self.q_proj(queries).view(b, l_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(keys).view(b, l_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(values).view(b, l_k, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = apply_attention_mask(scores, attn_mask)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, l_q, self.d_model)
        out = self.out_proj(out)
        return out, attn if self.output_attention else None


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class PCDEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, moving_avg=25, top_k=1,
                 dropout=0.1, activation='gelu', output_attention=False):
        super(PCDEncoderLayer, self).__init__()
        self.attention = PaperSparseMultiHeadAttention(d_model, n_heads, top_k=top_k,
                                                       dropout=dropout,
                                                       output_attention=output_attention)
        self.decomp1 = SeriesDecompLastDim(moving_avg)
        self.decomp2 = SeriesDecompLastDim(moving_avg)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x, _ = self.decomp1(x + self.dropout(new_x))
        y = self.norm1(x)
        x, _ = self.decomp2(y + self.dropout(self.ff(y)))
        x = self.norm2(x)
        return x, attn


class PCDEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(PCDEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class PCDDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, moving_avg=25, top_k=1,
                 dropout=0.1, activation='gelu'):
        super(PCDDecoderLayer, self).__init__()
        self.self_attention = PaperSparseMultiHeadAttention(d_model, n_heads, top_k=top_k,
                                                            dropout=dropout,
                                                            output_attention=False)
        self.cross_attention = PaperDenseMultiHeadAttention(d_model, n_heads,
                                                            dropout=dropout,
                                                            output_attention=False)
        self.decomp1 = SeriesDecompLastDim(moving_avg)
        self.decomp2 = SeriesDecompLastDim(moving_avg)
        self.decomp3 = SeriesDecompLastDim(moving_avg)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        new_x, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x, trend1 = self.decomp1(x + self.dropout(new_x))

        q = self.norm1(x)
        cross_out, _ = self.cross_attention(q, cross, cross, attn_mask=cross_mask)
        x, trend2 = self.decomp2(q + self.dropout(cross_out))

        y = self.norm2(x)
        x, trend3 = self.decomp3(y + self.dropout(self.ff(y)))
        x = self.norm3(x)
        trend = trend1 + trend2 + trend3
        return x, trend


class PCDDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(PCDDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        trend_sum = None
        for layer in self.layers:
            x, trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend_sum = trend if trend_sum is None else trend_sum + trend
        if self.norm is not None:
            x = self.norm(x)
        if trend_sum is None:
            trend_sum = torch.zeros_like(x)
        return x, trend_sum


class TrendModule(nn.Module):
    """
    Trend module in Eq. (8): FC(SD(X_enc) + Mean(X_enc)).

    Since SD returns seasonal and trend components, this implementation uses
    the trend component of SD(X_enc) and adds the mean sequence component.
    """
    def __init__(self, seq_len, d_model, moving_avg=25, dropout=0.0):
        super(TrendModule, self).__init__()
        self.decomp = SeriesDecompLastDim(moving_avg)
        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var_time):
        # x_var_time: [B, N, L]
        _, trend = self.decomp(x_var_time)
        mean = torch.mean(x_var_time, dim=-1, keepdim=True).repeat(1, 1, x_var_time.shape[-1])
        trend_feature = trend + mean
        return self.dropout(self.proj(trend_feature))


class Model(nn.Module):
    """
    PCDformer implemented in the Autoformer codebase with a paper-aligned
    architecture.

    Forward signature is kept compatible with Autoformer:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, ...)
    PCDformer itself treats each variable's full look-back window as one token,
    so time-mark embeddings are not used by this model.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.top_k = getattr(configs, 'top_k', 1)

        self.parallel_processing = ParallelProcessingLayer(
            seq_len=configs.seq_len,
            n_vars=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        self.dec_embedding = VariableEmbedding(
            input_len=configs.seq_len + configs.pred_len,
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        self.encoder = PCDEncoder(
            [
                PCDEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    top_k=self.top_k,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    output_attention=configs.output_attention
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.decoder = PCDDecoder(
            [
                PCDDecoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    top_k=self.top_k,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.trend_module = TrendModule(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            moving_avg=configs.moving_avg,
            dropout=configs.dropout
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [B, L, N]
        b, l, n = x_enc.shape
        assert l == self.seq_len, "x_enc length must equal configs.seq_len"
        assert n == self.enc_in, "x_enc variable count must equal configs.enc_in"

        # Algorithm 2, steps 2-3: parallel processing layer and encoder.
        enc_embed = self.parallel_processing(x_enc)       # [B, N, D]
        enc_out, attns = self.encoder(enc_embed, attn_mask=enc_self_mask)

        # Algorithm 2, step 4: X_dec = [observed history, zeros for future].
        zeros = torch.zeros(b, self.pred_len, n, device=x_enc.device, dtype=x_enc.dtype)
        dec_raw = torch.cat([x_enc, zeros], dim=1).permute(0, 2, 1).contiguous()
        dec_embed = self.dec_embedding(dec_raw)           # [B, N, D]

        # Algorithm 2, steps 5-8: decoder + trend module + FC + transpose.
        dec_out, trend_residual = self.decoder(dec_embed, enc_out,
                                               x_mask=dec_self_mask,
                                               cross_mask=dec_enc_mask)
        trend_out = self.trend_module(x_enc.permute(0, 2, 1).contiguous())
        out = dec_out + trend_out + trend_residual
        out = self.projection(out).transpose(1, 2).contiguous()  # [B, tau, N]

        if self.output_attention:
            return out, attns
        return out
