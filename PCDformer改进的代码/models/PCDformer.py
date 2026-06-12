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


def apply_attention_mask_if_compatible(scores, attn_mask):
    """
    ACR-Attention has two score shapes:
      variables -> cores: [B, H, N, R]
      cores -> variables: [B, H, R, N]

    Existing masks in this codebase are usually designed for square attention
    matrices. To avoid shape errors, this helper only applies a mask when the
    last two dimensions are compatible.
    """
    if attn_mask is None:
        return scores

    mask = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask

    if mask.shape[-2:] != scores.shape[-2:]:
        return scores

    return scores.masked_fill(mask, float('-inf'))


def sparsemax(logits, dim=-1):
    """
    Sparsemax activation.

    Compared with softmax, sparsemax can output exact zeros. This is useful for
    adaptive sparse routing because irrelevant cores/variables can be suppressed
    without manually selecting top-k.
    """
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, dim=dim, descending=True).values

    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.size(dim)
    rhos = torch.arange(
        1,
        logits.size(dim) + 1,
        device=logits.device,
        dtype=logits.dtype
    ).view(range_shape)

    cumsum_zs = torch.cumsum(zs, dim=dim)
    support = 1 + rhos * zs > cumsum_zs
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (torch.gather(cumsum_zs, dim, support_size.long() - 1) - 1) / support_size

    return torch.clamp(logits - tau, min=0.0)


def auto_core_num(n_vars):
    """
    Set the number of routing cores without any command-line switch.

    The purpose is not to mimic the original N x N top-k attention. The core
    count is chosen by variable scale:
      small-N datasets keep a few compact cores to avoid extra overhead;
      large-N datasets use more cores to keep enough cross-variable capacity.

    Examples:
      N=7   -> R=2
      N=21  -> R=4
      N=321 -> R=16
      N=862 -> R=30
    """
    n_vars = int(max(n_vars, 1))
    if n_vars <= 8:
        return 2
    if n_vars <= 32:
        return 4
    if n_vars <= 128:
        return 8
    if n_vars <= 512:
        return 16
    return min(32, int(math.ceil(math.sqrt(n_vars))))


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


class AdaptiveCoreRoutedSparseAttention(nn.Module):
    """
    Efficient Self-Preserved Adaptive Core-Routed Attention (SP-ACR).

    This version completely removes the previous direct top-k rescue branch.
    There is no original N x N top-k computation inside this module.

    The module keeps the useful idea of ACR:
        variables -> routing cores -> variables
    whose interaction cost is O(NR) instead of O(N^2), where R << N on large
    datasets.

    The key improvement over the previous ACR is a self-preserved local path:
    each variable keeps its own projected feature and only receives a gated
    routed-core message. This avoids over-compressing small-variable datasets
    such as ETT and Weather, while preserving the large-N acceleration on
    Traffic.

    Input/Output:
        queries/keys/values: [B, N, D]
        output:              [B, N, D]
    """
    def __init__(
        self,
        d_model,
        n_heads,
        n_vars,
        core_num=0,
        dropout=0.1,
        output_attention=False,
        route_activation='softmax',
        top_k=None,
        hybrid_direct_threshold=None
    ):
        super(AdaptiveCoreRoutedSparseAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_vars = int(n_vars)
        self.core_num = int(core_num) if int(core_num) > 0 else auto_core_num(n_vars)
        self.output_attention = output_attention
        self.scale = math.sqrt(d_model)

        # Learnable global cores. They are sample-adapted by the mean context.
        self.core_tokens = nn.Parameter(torch.randn(1, self.core_num, d_model) * 0.02)
        self.context_proj = nn.Linear(d_model, d_model)

        # Stage 1: cores absorb information from variables.
        self.core_q_proj = nn.Linear(d_model, d_model)
        self.var_k_proj = nn.Linear(d_model, d_model)
        self.var_v_proj = nn.Linear(d_model, d_model)

        # Stage 2: variables retrieve information from routed cores.
        self.var_q_proj = nn.Linear(d_model, d_model)
        self.core_k_proj = nn.Linear(d_model, d_model)
        self.core_v_proj = nn.Linear(d_model, d_model)

        # Self-preserved local path. This is not N x N attention; it is a
        # per-variable projection, so it keeps the original variable identity
        # with O(ND^2) linear cost and no pairwise variable score matrix.
        self.local_proj = nn.Linear(d_model, d_model)

        # Gated fusion between local feature and routed cross-variable message.
        # The bias is initialized negative so early training prefers the stable
        # local path, then learns to accept routed information when useful.
        self.fuse_gate = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Keep the local path close to identity at initialization. This makes
        # small-N datasets much less likely to be damaged by core compression.
        if self.local_proj.weight.shape[0] == self.local_proj.weight.shape[1]:
            nn.init.eye_(self.local_proj.weight)
        else:
            nn.init.xavier_uniform_(self.local_proj.weight)
        nn.init.zeros_(self.local_proj.bias)

        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        nn.init.xavier_uniform_(self.core_q_proj.weight)
        nn.init.zeros_(self.core_q_proj.bias)
        nn.init.xavier_uniform_(self.var_k_proj.weight)
        nn.init.zeros_(self.var_k_proj.bias)
        nn.init.xavier_uniform_(self.var_v_proj.weight)
        nn.init.zeros_(self.var_v_proj.bias)
        nn.init.xavier_uniform_(self.var_q_proj.weight)
        nn.init.zeros_(self.var_q_proj.bias)
        nn.init.xavier_uniform_(self.core_k_proj.weight)
        nn.init.zeros_(self.core_k_proj.bias)
        nn.init.xavier_uniform_(self.core_v_proj.weight)
        nn.init.zeros_(self.core_v_proj.bias)

        nn.init.xavier_uniform_(self.fuse_gate.weight, gain=0.5)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x):
        b, length, _ = x.shape
        x = x.view(b, length, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        b, _, length, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, length, self.d_model)

    def _attend(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = apply_attention_mask_if_compatible(scores, attn_mask)

        # Softmax is deliberately used here instead of sparsemax. The previous
        # sparsemax implementation requires sort/cumsum/gather and is often
        # slower than dense GPU matmul for small N. Core routing already limits
        # the interaction dimension to R, so exact-zero sparsemax is unnecessary
        # for real-time inference.
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, queries, keys, values, attn_mask=None):
        b, _, _ = queries.shape

        # Sample-adaptive cores: global learnable cores plus sample context.
        base_cores = self.core_tokens.expand(b, -1, -1)
        context = torch.tanh(self.context_proj(keys.mean(dim=1, keepdim=True)))
        core_queries = base_cores + context

        # Stage 1: cores absorb variable information. Shape cost: R x N.
        cq = self._split_heads(self.core_q_proj(core_queries))
        vk = self._split_heads(self.var_k_proj(keys))
        vv = self._split_heads(self.var_v_proj(values))
        routed_cores, _ = self._attend(cq, vk, vv, attn_mask=attn_mask)
        routed_cores = self._merge_heads(routed_cores)

        # Stage 2: variables retrieve core messages. Shape cost: N x R.
        vq = self._split_heads(self.var_q_proj(queries))
        ck = self._split_heads(self.core_k_proj(routed_cores))
        cv = self._split_heads(self.core_v_proj(routed_cores))
        routed_msg, var_to_core_attn = self._attend(vq, ck, cv, attn_mask=None)
        routed_msg = self._merge_heads(routed_msg)

        local_msg = self.local_proj(queries)
        route_gate = torch.sigmoid(self.fuse_gate(torch.cat([queries, routed_msg], dim=-1)))
        fused = (1.0 - route_gate) * local_msg + route_gate * routed_msg
        out = self.out_proj(fused)
        out = self.dropout(out)

        return out, var_to_core_attn if self.output_attention else None


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
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff=None,
        moving_avg=25,
        top_k=1,
        dropout=0.1,
        activation='gelu',
        output_attention=False,
        n_vars=None,
        attention_type='acr',
        core_num=0,
        route_activation='softmax',
        hybrid_direct_threshold=64
    ):
        super(PCDEncoderLayer, self).__init__()

        if n_vars is None:
            raise ValueError("n_vars must be provided for core-routed attention")

        self.attention = AdaptiveCoreRoutedSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_vars=n_vars,
            core_num=core_num,
            dropout=dropout,
            output_attention=output_attention,
            route_activation=route_activation,
            top_k=top_k,
            hybrid_direct_threshold=hybrid_direct_threshold
        )

        self.decomp1 = SeriesDecompLastDim(moving_avg)
        self.decomp2 = SeriesDecompLastDim(moving_avg)

        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

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
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff=None,
        moving_avg=25,
        top_k=1,
        dropout=0.1,
        activation='gelu',
        n_vars=None,
        attention_type='acr',
        core_num=0,
        route_activation='sparsemax',
        hybrid_direct_threshold=64
    ):
        super(PCDDecoderLayer, self).__init__()

        if attention_type == 'acr':
            if n_vars is None:
                raise ValueError("n_vars must be provided when attention_type='acr'")
            self.self_attention = AdaptiveCoreRoutedSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_vars=n_vars,
                core_num=core_num,
                dropout=dropout,
                output_attention=False,
                route_activation=route_activation,
                top_k=top_k,
                hybrid_direct_threshold=hybrid_direct_threshold
            )
        elif attention_type == 'topk':
            self.self_attention = PaperSparseMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                top_k=top_k,
                dropout=dropout,
                output_attention=False
            )
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # Keep decoder cross-attention dense for the first innovation.
        # This keeps the change focused on variable self-attention only.
        self.cross_attention = PaperDenseMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            output_attention=False
        )
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



class ResidualForecastAdapter(nn.Module):
    """
    Trend-Seasonal Residual Forecast Adapter (TS-RFA).

    This is not a dataset switch. It is a learnable prediction-space residual
    branch shared by all datasets. It adds a small DLinear-style trend/seasonal
    extrapolator beside PCDformer and learns whether the final forecast needs a
    direct temporal residual correction.

    Input:  x_var_time [B, L, N]
    Output: residual   [B, pred_len, N]
    """
    def __init__(self, seq_len, pred_len, n_vars, moving_avg=25, dropout=0.0, gate_init=-4.0):
        super(ResidualForecastAdapter, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.decomp = SeriesDecompLastDim(moving_avg)
        self.seasonal_proj = nn.Linear(seq_len, pred_len)
        self.trend_proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # Start as a very small residual corrector, not a disruptive branch.
        nn.init.xavier_uniform_(self.seasonal_proj.weight, gain=0.05)
        nn.init.zeros_(self.seasonal_proj.bias)
        nn.init.xavier_uniform_(self.trend_proj.weight, gain=0.05)
        nn.init.zeros_(self.trend_proj.bias)

    def forward(self, x_var_time):
        # x_var_time: [B, L, N] -> [B, N, L]
        x = x_var_time.permute(0, 2, 1).contiguous()
        seasonal, trend = self.decomp(x)
        residual = self.seasonal_proj(seasonal) + self.trend_proj(trend)
        residual = self.dropout(residual)
        return torch.sigmoid(self.gate) * residual.transpose(1, 2).contiguous()


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

        # attention_type='acr':  use Adaptive Core-Routed Sparse Attention.
        # attention_type='topk': use original fixed top-k sparse attention.
        self.attention_type = getattr(configs, 'attention_type', 'acr')
        self.core_num = getattr(configs, 'core_num', 0)
        self.route_activation = getattr(configs, 'route_activation', 'sparsemax')
        self.hybrid_direct_threshold = getattr(configs, 'hybrid_direct_threshold', 64)
        self.use_revin = getattr(configs, 'use_revin', 0) == 1
        self.revin_eps = getattr(configs, 'revin_eps', 1e-5)
        self.use_residual_adapter = getattr(configs, 'use_residual_adapter', 0) == 1
        self.adapter_gate_init = getattr(configs, 'adapter_gate_init', -4.0)

        if self.use_revin:
            self.revin_affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.revin_affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

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
                    output_attention=configs.output_attention,
                    n_vars=configs.enc_in,
                    attention_type=self.attention_type,
                    core_num=self.core_num,
                    route_activation=self.route_activation,
                    hybrid_direct_threshold=self.hybrid_direct_threshold
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
                    activation=configs.activation,
                    n_vars=configs.enc_in,
                    attention_type=self.attention_type,
                    core_num=self.core_num,
                    route_activation=self.route_activation,
                    hybrid_direct_threshold=self.hybrid_direct_threshold
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

        if self.use_residual_adapter:
            self.residual_adapter = ResidualForecastAdapter(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                n_vars=configs.enc_in,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                gate_init=self.adapter_gate_init
            )

        self.projection = nn.Linear(configs.d_model, configs.pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None
    ):
        # x_enc: [B, L, N]
        b, l, n = x_enc.shape
        assert l == self.seq_len, "x_enc length must equal configs.seq_len"
        assert n == self.enc_in, "x_enc variable count must equal configs.enc_in"

        x_raw = x_enc
        if self.use_revin:
            revin_mean = x_enc.mean(dim=1, keepdim=True).detach()
            revin_std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + self.revin_eps).detach()
            x_enc = (x_enc - revin_mean) / revin_std
            x_enc = x_enc * self.revin_affine_weight + self.revin_affine_bias

        # Algorithm 2, steps 2-3: parallel processing layer and encoder.
        enc_embed = self.parallel_processing(x_enc)       # [B, N, D]
        enc_out, attns = self.encoder(enc_embed, attn_mask=enc_self_mask)

        # Algorithm 2, step 4: X_dec = [observed history, zeros for future].
        zeros = torch.zeros(b, self.pred_len, n, device=x_enc.device, dtype=x_enc.dtype)
        dec_raw = torch.cat([x_enc, zeros], dim=1).permute(0, 2, 1).contiguous()
        dec_embed = self.dec_embedding(dec_raw)           # [B, N, D]

        # Algorithm 2, steps 5-8: decoder + trend module + FC + transpose.
        dec_out, trend_residual = self.decoder(
            dec_embed,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask
        )
        trend_out = self.trend_module(x_enc.permute(0, 2, 1).contiguous())
        out = dec_out + trend_out + trend_residual
        out = self.projection(out).transpose(1, 2).contiguous()  # [B, tau, N]

        if self.use_residual_adapter:
            out = out + self.residual_adapter(x_enc)

        if self.use_revin:
            out = (out - self.revin_affine_bias) / (self.revin_affine_weight + self.revin_eps)
            out = out * revin_std + revin_mean

        if self.output_attention:
            return out, attns
        return out