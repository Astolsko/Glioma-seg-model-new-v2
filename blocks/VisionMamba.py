"""SegMamba-style Vision Mamba encoder for 3D brain-tumour segmentation.

Swappable replacement for the ViT encoder in blocks/Transformer.py, selected
with `cfg.unetr.encoder = "mamba"` (see models/unetr.py). The decoder, losses,
training schedule and inference recipe are shared with the ViT model, so the
encoder is the only thing that differs between the two runs.

Reference
---------
Xing, Ye, Yang, Ning, Zhu. "SegMamba: Long-range Sequential Modeling Mamba For
3D Medical Image Segmentation." MICCAI 2024, arXiv:2401.13560. Evaluated on
BraTS 2023. Official code: https://github.com/ge-xing/SegMamba
  model_segmamba/segmamba.py                  MambaEncoder, MambaLayer, GSC, MlpChannel
  mamba/mamba_ssm/modules/mamba_simple.py     the tri-orientated Mamba (ToM) mixer,
  (bimamba_type="v3")                         i.e. TriOrientedMamba below

Reproduced exactly from the official code
-----------------------------------------
  stem           Conv3d(in, 48, k=7, s=2, p=3)
  downsample     InstanceNorm3d -> Conv3d(k=2, s=2), between consecutive stages
  stage i        GSC -> depth_i x MambaLayer, dims 48/96/192/384, depth 2 each
  stage output   InstanceNorm3d -> MlpChannel (1x1x1 conv, GELU, 1x1x1 conv, 2x hidden)
  MambaLayer     LayerNorm over channels -> ToM -> + residual
  ToM            a shared in_proj (d -> 2*d_inner, no bias) feeds three
                 independent selective-scan branches (each with its own causal
                 depthwise conv1d, x_proj, dt_proj, A_log, D), run on three
                 orderings of the flattened volume. Branch keys map to the
                 official parameter suffixes: "fwd" = none, "bwd" = _b, "slice" = _s.
                   fwd    raster order (D fastest, then W, then H)
                   bwd    that sequence reversed
                   slice  interleaved across the H-slabs, i.e. a scan along
                          H for every (w, d): SegMamba's inter-slice path
                 The branch outputs are mapped back to raster order and summed
                 before one shared out_proj (d_inner -> d, no bias).
  init           S4D-real A_log, D = 1 in every branch. Mamba's special dt
                 initialisation on the FORWARD branch only; the backward and
                 slice branches keep nn.Linear's default init for dt_proj,
                 because that is what the official v3 code does (only
                 self.dt_proj is specially initialised there).
  nslices        SegMamba hard-codes [64, 32, 16, 8] for 128^3 inputs, which
                 is the H extent of each stage. Here it is read off the
                 feature map, which gives the same numbers for our 128x128x96
                 patches (H = 128) and stays correct for any other size.

Deliberate deviations, each needed for this study
-------------------------------------------------
  1. Dropout. SegMamba has none. The ViT carries p=0.2 dropout in every block,
     and the MC-dropout uncertainty XAI samples dropout layers, so without
     them the Mamba model's uncertainty would come from three decoder layers
     only and the two encoders' maps would not be comparable. The rate is
     cfg.mamba.dropout. It sits after each ToM, before the residual add (the
     analogue of the ViT's attention proj_dropout), and inside each
     MlpChannel after the GELU (the analogue of the ViT's MLP dropout).
  2. Kernel packaging. The official code ships a forked mamba_ssm whose
     `mamba_inner_fn_no_out_proj` returns the pre-projection scan output. Here
     each branch calls upstream mamba_ssm's fused `mamba_inner_fn`, which
     applies out_proj itself. out_proj has no bias and acts per token, so
         out_proj(y_f + unflip(y_b) + unslice(y_s))
       = out_proj(y_f) + unflip(out_proj(y_b)) + unslice(out_proj(y_s)).
     It is the same function at the cost of two extra per-token matmuls;
     tests/test_vision_mamba.py checks the fused path against the reference.
  3. Fallback. Without the CUDA kernels the module uses an exact chunked
     reference scan, practical only for the tiny test models. On a CUDA
     tensor without the kernels it raises rather than silently crawling
     through a 196k-token volume.

Explainability
--------------
`TriOrientedMamba.hidden_attention` returns the layer's implicit attention
matrix (Ali, Zimerman & Wolf, "The Hidden Attention of Mamba Models",
arXiv:2403.01590). Unrolling the selective-scan recurrence
    h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t,     y_t = C_t h_t + D u_t
gives y_t = sum_{s<=t} alpha[t, s] u_s + D u_t, with
    alpha[t, s] = sum_n C_t[n] exp(sum_{k=s+1..t} dt_k A[n]) dt_s B_s[n]
per channel. |alpha| is averaged over channels, each branch is mapped back to
raster order and the three are summed: that is the token-mixing operator the
layer applies, up to the local conv1d and the SiLU output gate, which Ali et
al.'s alpha also leave out.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn as _mamba_inner_fn
except (ImportError, OSError):
    # No kernels (e.g. the pytorch2 env, or a CPU-only test run): the
    # reference path below still works on CPU, and a CUDA forward refuses.
    _mamba_inner_fn = None


DIRECTIONS = ("fwd", "bwd", "slice")


def mamba_kernels_available():
    """True when mamba_ssm's fused CUDA kernels are importable."""
    return _mamba_inner_fn is not None


# ---------------------------------------------------------------------------
# Selective scan: reference implementations
# ---------------------------------------------------------------------------

def _prepare_delta(delta, delta_bias, delta_softplus):
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    return delta


def _finish(y, u, D, z, dtype):
    if D is not None:
        y = y + u.float() * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype)


def selective_scan_sequential(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                              delta_softplus=True):
    """The selective-scan recurrence, one step at a time. The definition the
    faster implementations are tested against, not something to train with.

    u, delta, z: (batch, d, L). A: (d, n). B, C: (batch, n, L). D: (d,).
    """
    dtype = u.dtype
    delta = _prepare_delta(delta, delta_bias, delta_softplus)
    u32, A, B, C = u.float(), A.float(), B.float(), C.float()
    h = u32.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for t in range(u.shape[-1]):
        decay = torch.exp(delta[:, :, t, None] * A)                       # (b, d, n)
        h = decay * h + (delta[:, :, t] * u32[:, :, t])[..., None] * B[:, None, :, t]
        ys.append((h * C[:, None, :, t]).sum(-1))
    return _finish(torch.stack(ys, dim=-1), u, D, z, dtype)


def _decay_matrix(S):
    """exp(S_t - S_s) for s <= t and 0 above the diagonal, from the running
    log-decay S of shape (b, d, T, n). Returns (b, d, T, T, n), index [t, s].
    The mask goes in BEFORE the exp so the s > t entries never overflow."""
    T = S.shape[2]
    causal = torch.ones(T, T, dtype=torch.bool, device=S.device).tril()
    diff = S[:, :, :, None, :] - S[:, :, None, :, :]
    return torch.exp(diff.masked_fill(~causal[None, None, :, :, None], float("-inf")))


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=True, chunk=64):
    """Exact selective scan, vectorised within chunks of `chunk` steps and
    carrying the state between chunks. The CPU/test path; same arguments and
    result as selective_scan_sequential."""
    dtype = u.dtype
    delta = _prepare_delta(delta, delta_bias, delta_softplus)
    u32, A, B, C = u.float(), A.float(), B.float(), C.float()
    h = u32.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for start in range(0, u.shape[-1], chunk):
        end = min(start + chunk, u.shape[-1])
        dl = delta[:, :, start:end]                                        # (b, d, T)
        S = torch.cumsum(dl[..., None] * A[None, :, None, :], dim=2)      # (b, d, T, n)
        bu = (dl * u32[:, :, start:end])[..., None] * B[:, None, :, start:end].transpose(-1, -2)
        states = torch.einsum("bdtsn,bdsn->bdtn", _decay_matrix(S), bu)
        states = states + torch.exp(S) * h[:, :, None, :]
        ys.append(torch.einsum("bdtn,bnt->bdt", states, C[:, :, start:end]))
        h = states[:, :, -1]
    return _finish(torch.cat(ys, dim=-1), u, D, z, dtype)


def hidden_attention_matrix(delta, A, B, C, delta_bias=None, delta_softplus=True,
                            reduce=True, channel_chunk=16):
    """Implicit attention of a selective scan (Ali et al. 2024).

    Returns alpha of shape (batch, d, L, L) with reduce=False, or the channel
    mean of |alpha|, shape (batch, L, L), with reduce=True. Index [t, s] is
    how much input token s contributes to output token t (zero for s > t).
    Channels are processed in chunks so an (L, L, n) block per channel never
    has to exist for all d at once.
    """
    delta = _prepare_delta(delta, delta_bias, delta_softplus)
    A, B, C = A.float(), B.float(), C.float()
    batch, d, L = delta.shape
    acc = delta.new_zeros(batch, L, L) if reduce else None
    parts = []
    for c0 in range(0, d, channel_chunk):
        c1 = min(c0 + channel_chunk, d)
        dl = delta[:, c0:c1]
        S = torch.cumsum(dl[..., None] * A[None, c0:c1, None, :], dim=2)
        alpha = torch.einsum("bdtsn,bnt,bns->bdts", _decay_matrix(S), C, B)
        alpha = alpha * dl[:, :, None, :]                                  # dt_s
        if reduce:
            acc += alpha.abs().sum(1)
        else:
            parts.append(alpha)
    return acc / d if reduce else torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Scan orderings (SegMamba ToM)
# ---------------------------------------------------------------------------

def to_scan_order(seq, direction, nslices):
    """(batch, channels, L) in raster order -> the order `direction` scans.

    "slice" reproduces SegMamba exactly: chunk the flat sequence into nslices
    contiguous pieces (one per H index), stack them on a new last axis and
    flatten, so position i*nslices + j holds raster token j*(L/nslices) + i.
    """
    if direction == "fwd":
        return seq
    if direction == "bwd":
        return seq.flip(-1)
    if direction == "slice":
        batch, channels, L = seq.shape
        return (seq.reshape(batch, channels, nslices, L // nslices)
                .transpose(-1, -2).reshape(batch, channels, L))
    raise ValueError(f"unknown scan direction {direction!r}")


def from_scan_order(seq, direction, nslices):
    """Inverse of to_scan_order for (batch, L, channels) outputs (tokens on
    dim 1). For "slice" this is SegMamba's reshape/permute/flatten undo."""
    if direction == "fwd":
        return seq
    if direction == "bwd":
        return seq.flip(1)
    if direction == "slice":
        batch, L, channels = seq.shape
        return (seq.reshape(batch, L // nslices, nslices, channels)
                .transpose(1, 2).reshape(batch, L, channels))
    raise ValueError(f"unknown scan direction {direction!r}")


def attention_to_raster_order(attn, direction, nslices):
    """Map a (batch, L, L) attention matrix indexed in `direction`'s scan
    order to raster order on both axes."""
    L = attn.shape[-1]
    index = torch.arange(L, device=attn.device).view(1, 1, L)
    perm = to_scan_order(index, direction, nslices).view(L)   # scan pos -> raster idx
    raster = torch.empty_like(attn)
    raster[:, perm[:, None], perm[None, :]] = attn
    return raster


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SelectiveSSMBranch(nn.Module):
    """One selective-scan (S6) path of the ToM mixer.

    causal depthwise conv1d -> SiLU -> input-dependent (dt, B, C) -> scan,
    gated by SiLU(z). Parameter layout is Mamba's own: conv1d, x_proj,
    dt_proj, A_log, D. SegMamba names these conv1d/conv1d_b/conv1d_s etc. on
    one module; here each branch is its own module.
    """

    def __init__(self, d_inner, d_state, d_conv, dt_rank, mamba_dt_init,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_scale=1.0):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.mamba_dt_init = mamba_dt_init
        self.dt_min, self.dt_max = dt_min, dt_max
        self.dt_init_floor, self.dt_scale = dt_init_floor, dt_scale

        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.D = nn.Parameter(torch.empty(d_inner))
        # Mamba convention: the state decay and the skip are not weight-decayed
        # (utils/engine.py puts these in a weight_decay=0 AdamW group).
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.reset_parameters()

    def reset_parameters(self):
        """Mamba's SSM initialisation: S4D-real A, D = 1 and, on the branch
        that gets it, the dt init that puts softplus(dt_bias) log-uniformly
        in [dt_min, dt_max]. conv1d/x_proj keep their own default init, as in
        Mamba. Also what the XAI randomisation test calls to re-draw weights,
        after the children have reset themselves (see utils/xai.py)."""
        with torch.no_grad():
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32,
                             device=self.A_log.device).repeat(self.d_inner, 1)
            self.A_log.copy_(torch.log(A))
            self.D.fill_(1.0)
            if self.mamba_dt_init:
                std = self.dt_rank ** -0.5 * self.dt_scale
                nn.init.uniform_(self.dt_proj.weight, -std, std)
                dt = torch.exp(
                    torch.rand(self.d_inner, device=self.dt_proj.bias.device)
                    * (math.log(self.dt_max) - math.log(self.dt_min))
                    + math.log(self.dt_min)
                ).clamp(min=self.dt_init_floor)
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1

    def scan_inputs(self, xz):
        """(u, z, delta, B, C) exactly as mamba_inner_ref computes them.
        xz is (batch, 2*d_inner, L) in THIS branch's scan order; delta is
        returned without its bias, which the scan adds."""
        L = xz.shape[-1]
        x, z = xz.chunk(2, dim=1)
        u = F.silu(F.conv1d(x, self.conv1d.weight, self.conv1d.bias,
                            padding=self.d_conv - 1, groups=self.d_inner)[..., :L])
        x_dbl = F.linear(u.transpose(1, 2), self.x_proj.weight)            # (b, L, r+2n)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2)           # (b, d, L)
        return u, z, delta, B.transpose(1, 2), C.transpose(1, 2)

    def forward_reference(self, xz):
        """Pre-projection output y, (batch, d_inner, L), without the kernels."""
        u, z, delta, B, C = self.scan_inputs(xz)
        A = -torch.exp(self.A_log.float())
        return selective_scan_chunked(u, delta, A, B, C, self.D, z=z,
                                      delta_bias=self.dt_proj.bias, delta_softplus=True)

    def hidden_attention(self, xz, reduce=True):
        """Implicit attention of this branch, in its scan order."""
        _, _, delta, B, C = self.scan_inputs(xz)
        A = -torch.exp(self.A_log.float())
        return hidden_attention_matrix(delta, A, B, C, delta_bias=self.dt_proj.bias,
                                       delta_softplus=True, reduce=reduce)


class TriOrientedMamba(nn.Module):
    """SegMamba's tri-orientated Mamba (ToM) mixer (bimamba_type="v3").

    forward(hidden, nslices): hidden is (batch, L, d_model) tokens in raster
    order of an (H, W, D) grid and nslices = H. Returns (batch, L, d_model).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.branches = nn.ModuleDict({
            direction: SelectiveSSMBranch(
                self.d_inner, d_state, d_conv, self.dt_rank,
                mamba_dt_init=(direction == "fwd"),
            )
            for direction in DIRECTIONS
        })
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _resolve_backend(x, backend):
        if backend is not None:
            return backend
        if x.is_cuda:
            if not mamba_kernels_available():
                raise RuntimeError(
                    "cfg.unetr.encoder='mamba' on a CUDA tensor needs the mamba_ssm "
                    "CUDA kernels, which are not importable in this env. Run from "
                    "the `mamba` conda env (conda activate mamba).")
            return "cuda"
        return "reference"

    def _run_branch(self, branch, xz, backend):
        """One branch, projected: (batch, L, d_model) in the branch's scan order."""
        if backend == "cuda":
            A = -torch.exp(branch.A_log.float())
            return _mamba_inner_fn(
                xz, branch.conv1d.weight, branch.conv1d.bias, branch.x_proj.weight,
                branch.dt_proj.weight, self.out_proj.weight, self.out_proj.bias,
                A, None, None, branch.D.float(),
                delta_bias=branch.dt_proj.bias.float(), delta_softplus=True,
            )
        y = branch.forward_reference(xz)
        return F.linear(y.transpose(1, 2), self.out_proj.weight, self.out_proj.bias)

    def forward(self, hidden, nslices, backend=None):
        L = hidden.shape[1]
        if L % nslices:
            raise ValueError(f"sequence length {L} is not divisible by nslices={nslices}")
        backend = self._resolve_backend(hidden, backend)
        xz = self.in_proj(hidden).transpose(1, 2).contiguous()             # (b, 2*d_inner, L)
        out = None
        for direction, branch in self.branches.items():
            y = self._run_branch(branch, to_scan_order(xz, direction, nslices).contiguous(),
                                 backend)
            y = from_scan_order(y, direction, nslices)
            out = y if out is None else out + y
        return out

    @torch.no_grad()
    def hidden_attention(self, hidden, nslices):
        """(batch, L, L) channel-mean |alpha| summed over the three branches,
        both axes in raster order. [t, s] = how much token s feeds token t."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            xz = F.linear(hidden, self.in_proj.weight.float()).transpose(1, 2).contiguous()
            total = None
            for direction, branch in self.branches.items():
                attn = branch.hidden_attention(to_scan_order(xz, direction, nslices))
                attn = attention_to_raster_order(attn, direction, nslices)
                total = attn if total is None else total + attn
        return total


class MambaLayer(nn.Module):
    """SegMamba MambaLayer: LayerNorm -> ToM -> residual, on a 3D feature map.
    The dropout before the residual add is deviation 1 in the module docstring."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = TriOrientedMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, channels = x.shape[:2]
        grid = x.shape[2:]
        if channels != self.dim:
            raise ValueError(f"MambaLayer(dim={self.dim}) got {channels} channels")
        tokens = x.reshape(batch, channels, -1).transpose(-1, -2)         # (b, L, C), raster
        # nslices is passed positionally so forward hooks see it in `inputs`.
        mixed = self.dropout(self.mamba(self.norm(tokens), grid[0]))
        return mixed.transpose(-1, -2).reshape(batch, channels, *grid) + x


class MlpChannel(nn.Module):
    """SegMamba MlpChannel (1x1x1 conv MLP), plus deviation-1 dropout."""

    def __init__(self, hidden_size, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class GSC(nn.Module):
    """SegMamba gated spatial convolution, verbatim: a 3x3x3 -> 3x3x3 branch
    and a 1x1x1 branch, summed, a 1x1x1 fuse, plus the input."""

    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()
        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()
        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        x2 = self.nonliner3(self.norm3(self.proj3(x)))
        out = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
        return out + x


class VisionMambaEncoder(nn.Module):
    """SegMamba's MambaEncoder. Returns four feature maps at 1/2, 1/4, 1/8 and
    1/16 of the input resolution with dims[0..3] channels."""

    def __init__(self, in_chans=4, dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
                 d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        dims, depths = tuple(dims), tuple(depths)
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("SegMamba has exactly four stages: dims/depths need 4 entries")
        self.dims = dims

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)),
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.gscs = nn.ModuleList([GSC(d) for d in dims])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                MambaLayer(dims[i], d_state=d_state, d_conv=d_conv, expand=expand,
                           dropout=dropout)
                for _ in range(depths[i])
            ])
            for i in range(4)
        ])
        for i in range(4):
            self.add_module(f"norm{i}", nn.InstanceNorm3d(dims[i]))
        self.mlps = nn.ModuleList([MlpChannel(d, 2 * d, dropout=dropout) for d in dims])

    def last_stage_mixers(self):
        """The ToM modules of the deepest stage, whose token grid is the same
        1/16 grid the ViT's patches live on. Used by the XAI hooks."""
        return [layer.mamba for layer in self.stages[-1]]

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            outs.append(self.mlps[i](getattr(self, f"norm{i}")(x)))
        return tuple(outs)
