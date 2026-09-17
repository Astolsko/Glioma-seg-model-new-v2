"""Tests for the SegMamba-style Vision Mamba encoder (blocks/VisionMamba.py)
and everything the encoder switch touches: the UNETR wiring, the optimizer
groups, checkpoint and run-snapshot handling, and the XAI + attention hooks.

Most tests run on CPU with the exact reference scan, so they pass in either
conda env. Tests that compare against mamba_ssm's fused CUDA kernels skip
unless a GPU and the kernels are present, i.e. run them from the `mamba` env:

    conda activate mamba && pytest tests/test_vision_mamba.py -m "slow or not slow"
"""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from blocks import VisionMamba as vm  # noqa: E402
from blocks.VisionMamba import (  # noqa: E402
    DIRECTIONS, TriOrientedMamba, VisionMambaEncoder, attention_to_raster_order,
    from_scan_order, hidden_attention_matrix, mamba_kernels_available,
    selective_scan_chunked, selective_scan_sequential, to_scan_order,
)

needs_kernels = pytest.mark.skipif(
    not (torch.cuda.is_available() and mamba_kernels_available()),
    reason="needs a GPU and the mamba_ssm CUDA kernels (run from the `mamba` env)")

VIT_CHILDREN = {
    "transformer", "dilated_bottleneck", "decoder0", "decoder3", "decoder6", "decoder9",
    "decoder12_upsampler", "decoder9_upsampler", "decoder6_upsampler",
    "decoder3_upsampler", "decoder0_header", "aux_head_z6", "aux_head_z3",
    "skip9", "skip6", "skip3",
}


def _scan_args(batch=2, d=6, L=50, n=4, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, d, L, generator=g)
    delta = torch.randn(batch, d, L, generator=g) * 0.5
    A = -torch.rand(d, n, generator=g) * 2 - 0.1
    B = torch.randn(batch, n, L, generator=g)
    C = torch.randn(batch, n, L, generator=g)
    D = torch.randn(d, generator=g)
    z = torch.randn(batch, d, L, generator=g)
    bias = torch.randn(d, generator=g) * 0.1
    out = [u, delta, A, B, C, D, z, bias]
    if requires_grad:
        out = [t.requires_grad_() for t in out]
    return out


def _rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# ---------------------------------------------------------------------------
# the selective scan and its hidden attention
# ---------------------------------------------------------------------------

def test_chunked_scan_matches_the_step_by_step_definition():
    u, delta, A, B, C, D, z, bias = _scan_args()
    ref = selective_scan_sequential(u, delta, A, B, C, D, z=z, delta_bias=bias)
    out = selective_scan_chunked(u, delta, A, B, C, D, z=z, delta_bias=bias, chunk=16)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_chunked_scan_gradients_match_the_definition():
    a = _scan_args(L=30, requires_grad=True)
    b = [t.detach().clone().requires_grad_() for t in a]
    selective_scan_sequential(*a[:6], z=a[6], delta_bias=a[7]).square().sum().backward()
    selective_scan_chunked(*b[:6], z=b[6], delta_bias=b[7], chunk=8).square().sum().backward()
    for ta, tb in zip(a, b):
        assert _rel_err(tb.grad, ta.grad) < 1e-4


def test_chunked_scan_matches_upstream_mamba_reference():
    iface = pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    u, delta, A, B, C, D, z, bias = _scan_args()
    ref = iface.selective_scan_ref(u, delta, A, B, C, D, z=z, delta_bias=bias,
                                   delta_softplus=True)
    out = selective_scan_chunked(u, delta, A, B, C, D, z=z, delta_bias=bias)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_hidden_attention_reproduces_the_scan_output():
    """Ali et al.'s alpha IS the scan: y_t = sum_s alpha[t, s] u_s exactly."""
    u, delta, A, B, C, _, _, bias = _scan_args(batch=1, L=40)
    alpha = hidden_attention_matrix(delta, A, B, C, delta_bias=bias, reduce=False)
    assert alpha.shape == (1, 6, 40, 40)
    assert torch.all(alpha.triu(1) == 0), "a causal scan cannot look ahead"
    y_alpha = torch.einsum("bdts,bds->bdt", alpha, u)
    y_scan = selective_scan_sequential(u, delta, A, B, C, D=None, z=None, delta_bias=bias)
    torch.testing.assert_close(y_alpha, y_scan, rtol=1e-4, atol=1e-5)


def test_reduced_hidden_attention_is_the_channel_mean_of_abs_alpha():
    _, delta, A, B, C, _, _, bias = _scan_args(batch=1, L=20)
    full = hidden_attention_matrix(delta, A, B, C, delta_bias=bias, reduce=False)
    reduced = hidden_attention_matrix(delta, A, B, C, delta_bias=bias, reduce=True,
                                      channel_chunk=4)          # 6 channels, uneven chunks
    torch.testing.assert_close(reduced, full.abs().mean(1))


# ---------------------------------------------------------------------------
# the three SegMamba scan orders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", DIRECTIONS)
def test_scan_order_round_trips(direction):
    H, W, D = 4, 3, 2
    seq = torch.randn(2, 5, H * W * D)
    scanned = to_scan_order(seq, direction, H)
    back = from_scan_order(scanned.transpose(1, 2), direction, H).transpose(1, 2)
    assert torch.equal(back, seq)


def test_slice_scan_walks_along_the_first_spatial_axis():
    H, W, D = 4, 3, 2
    order = to_scan_order(torch.arange(H * W * D).view(1, 1, -1), "slice", H).view(-1)
    expected = torch.arange(H * W * D).view(H, W, D).permute(1, 2, 0).reshape(-1)
    assert torch.equal(order, expected)       # H fastest: an inter-slice scan


def test_slice_order_is_segmambas_chunk_stack_flatten():
    """Literal transcription of the official v3 code, both directions."""
    n, L = 4, 24
    xz = torch.randn(2, 6, L)
    segmamba_in = torch.stack(xz.chunk(n, dim=-1), dim=-1).flatten(-2)
    assert torch.equal(to_scan_order(xz, "slice", n), segmamba_in)

    out = torch.randn(2, 6, L)
    segmamba_out = out.reshape(2, 6, L // n, n).permute(0, 1, 3, 2).flatten(-2)
    ours = from_scan_order(out.transpose(1, 2), "slice", n).transpose(1, 2)
    assert torch.equal(ours, segmamba_out)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_attention_maps_to_raster_order_consistently(direction):
    """If y = M x in scan order, then y = M_raster x in raster order."""
    H, L = 4, 24
    M = torch.randn(1, L, L)
    x = torch.randn(1, 1, L)
    y_scan = M @ to_scan_order(x, direction, H).view(1, L, 1)
    y = from_scan_order(y_scan, direction, H)
    M_raster = attention_to_raster_order(M, direction, H)
    torch.testing.assert_close(M_raster @ x.view(1, L, 1), y)


# ---------------------------------------------------------------------------
# the tri-orientated Mamba mixer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", DIRECTIONS)
def test_each_branch_is_causal_in_its_own_scan_order(direction):
    torch.manual_seed(0)
    tom = TriOrientedMamba(d_model=8, d_state=4)
    L, t = 16, 7
    xz = torch.randn(1, 2 * tom.d_inner, L, requires_grad=True)
    y = tom._run_branch(tom.branches[direction], xz, "reference")
    y[:, t].sum().backward()
    grad = xz.grad[0].abs().sum(0)
    assert torch.all(grad[t + 1:] == 0)
    assert grad[t] > 0


def test_tri_oriented_mamba_sees_the_whole_volume():
    """The backward and slice scans make every token visible to every other,
    unlike a single causal scan."""
    torch.manual_seed(0)
    tom = TriOrientedMamba(d_model=8, d_state=4)
    x = torch.randn(1, 2 * 3 * 2, 8, requires_grad=True)       # grid (2, 3, 2)
    tom(x, 2, backend="reference")[:, 0].sum().backward()
    assert torch.all(x.grad[0].abs().sum(-1) > 0)


def test_per_branch_projection_equals_segmambas_project_after_sum():
    """Deviation 2 in the module docstring: projecting each branch then
    summing is the same function as SegMamba's sum-then-project."""
    torch.manual_seed(0)
    tom = TriOrientedMamba(d_model=8, d_state=4)
    H = 4
    x = torch.randn(2, H * 3 * 2, 8)
    xz = tom.in_proj(x).transpose(1, 2)
    summed = 0
    for direction, branch in tom.branches.items():
        y = branch.forward_reference(to_scan_order(xz, direction, H).contiguous())
        summed = summed + from_scan_order(y.transpose(1, 2), direction, H)
    torch.testing.assert_close(tom(x, H, backend="reference"), tom.out_proj(summed))


def test_only_the_forward_branch_gets_mambas_dt_init():
    """Matches the official v3 code: only self.dt_proj is specially initialised."""
    tom = TriOrientedMamba(d_model=48)
    dt_forward = F.softplus(tom.branches["fwd"].dt_proj.bias)
    assert dt_forward.min() >= 1e-3 - 1e-6 and dt_forward.max() <= 0.1 + 1e-6
    for name in ("bwd", "slice"):
        assert F.softplus(tom.branches[name].dt_proj.bias).min() > 0.1


@needs_kernels
def test_fused_cuda_path_matches_the_reference_path():
    torch.manual_seed(0)
    tom = TriOrientedMamba(d_model=16, d_state=16).cuda()
    x = torch.randn(2, 8 * 4 * 2, 16, device="cuda")            # grid (8, 4, 2)
    x_cuda, x_ref = x.clone().requires_grad_(), x.clone().requires_grad_()

    out_cuda = tom(x_cuda, 8, backend="cuda")
    out_cuda.square().sum().backward()
    grads_cuda = {n: p.grad.clone() for n, p in tom.named_parameters()}
    tom.zero_grad()
    out_ref = tom(x_ref, 8, backend="reference")
    out_ref.square().sum().backward()

    assert _rel_err(out_cuda, out_ref) < 1e-4
    assert _rel_err(x_cuda.grad, x_ref.grad) < 1e-3
    for name, p in tom.named_parameters():
        assert _rel_err(grads_cuda[name], p.grad) < 1e-3, name


@needs_kernels
def test_fused_cuda_path_runs_under_bf16_autocast():
    torch.manual_seed(0)
    tom = TriOrientedMamba(d_model=16).cuda()
    x = torch.randn(2, 64, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = tom(x, 4)
    out.float().sum().backward()
    assert torch.isfinite(out.float()).all() and torch.isfinite(x.grad).all()


def test_cuda_forward_without_kernels_refuses(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    monkeypatch.setattr(vm, "_mamba_inner_fn", None)
    tom = TriOrientedMamba(8).cuda()
    with pytest.raises(RuntimeError, match="mamba"):
        tom(torch.randn(1, 8, 8, device="cuda"), 2)


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------

def test_encoder_returns_four_stages_at_segmamba_resolutions():
    torch.manual_seed(0)
    enc = VisionMambaEncoder(in_chans=4, dims=(8, 16, 32, 64), depths=(1, 1, 1, 2), d_state=4)
    outs = enc(torch.randn(1, 4, 32, 32, 16))
    assert [tuple(o.shape) for o in outs] == [
        (1, 8, 16, 16, 8), (1, 16, 8, 8, 4), (1, 32, 4, 4, 2), (1, 64, 2, 2, 1)]


def test_real_config_encoder_matches_the_segmamba_reference_layout():
    """Pins the published SegMamba architecture (segmamba.py + mamba_simple.py v3)."""
    from config import cfg
    assert list(cfg.mamba.dims) == [48, 96, 192, 384]
    assert list(cfg.mamba.depths) == [2, 2, 2, 2]
    assert (cfg.mamba.d_state, cfg.mamba.d_conv, cfg.mamba.expand) == (16, 4, 2)
    enc = VisionMambaEncoder(in_chans=4, dims=cfg.mamba.dims, depths=cfg.mamba.depths,
                             d_state=cfg.mamba.d_state, d_conv=cfg.mamba.d_conv,
                             expand=cfg.mamba.expand, dropout=cfg.mamba.dropout)

    stem = enc.downsample_layers[0][0]
    assert (stem.in_channels, stem.out_channels) == (4, 48)
    assert (stem.kernel_size, stem.stride, stem.padding) == ((7, 7, 7), (2, 2, 2), (3, 3, 3))
    for i in range(1, 4):
        norm, conv = enc.downsample_layers[i]
        assert isinstance(norm, torch.nn.InstanceNorm3d)
        assert conv.kernel_size == (2, 2, 2) and conv.stride == (2, 2, 2)

    for i, (dim, depth) in enumerate(zip(cfg.mamba.dims, cfg.mamba.depths)):
        assert len(enc.stages[i]) == depth
        assert isinstance(getattr(enc, f"norm{i}"), torch.nn.InstanceNorm3d)
        assert enc.mlps[i].fc1.out_channels == 2 * dim
        d_inner, rank = 2 * dim, math.ceil(dim / 16)
        for layer in enc.stages[i]:
            tom = layer.mamba
            assert tom.in_proj.weight.shape == (2 * d_inner, dim) and tom.in_proj.bias is None
            assert tom.out_proj.weight.shape == (dim, d_inner) and tom.out_proj.bias is None
            assert set(tom.branches) == {"fwd", "bwd", "slice"}
            for branch in tom.branches.values():
                assert branch.conv1d.weight.shape == (d_inner, 1, 4)
                assert branch.conv1d.groups == d_inner and branch.conv1d.bias is not None
                assert branch.x_proj.weight.shape == (rank + 32, d_inner)
                assert branch.dt_proj.weight.shape == (d_inner, rank)
                torch.testing.assert_close(
                    branch.A_log, torch.log(torch.arange(1, 17.0).repeat(d_inner, 1)))
                assert torch.all(branch.D == 1)


# ---------------------------------------------------------------------------
# the UNETR with the Mamba encoder
# ---------------------------------------------------------------------------

def test_mamba_unetr_train_forward_shapes_and_every_parameter_learns(tiny_mamba_unetr,
                                                                    tiny_mamba_kwargs):
    shape = tiny_mamba_kwargs["img_shape"]
    model = tiny_mamba_unetr.train()
    out, aux_z6, aux_z3 = model(torch.randn(1, 4, *shape))
    for t in (out, aux_z6, aux_z3):
        assert t.shape == (1, 3, *shape) and torch.isfinite(t).all()
    (out.mean() + aux_z6.mean() + aux_z3.mean()).backward()

    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not no_grad
    dead = [n for n, p in model.mamba_encoder.named_parameters() if p.grad.abs().sum() == 0]
    assert not dead, f"encoder parameters with an all-zero gradient: {dead[:5]}"


def test_mamba_unetr_eval_returns_logits_only(tiny_mamba_unetr, tiny_mamba_kwargs):
    shape = tiny_mamba_kwargs["img_shape"]
    with torch.no_grad():
        out = tiny_mamba_unetr.eval()(torch.randn(1, 4, *shape))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 3, *shape)


def test_vit_model_keeps_its_pre_switch_modules(tiny_unetr):
    """Old ViT checkpoints (v2-run3 etc.) must keep loading."""
    assert tiny_unetr.encoder_type == "vit"
    assert {name for name, _ in tiny_unetr.named_children()} == VIT_CHILDREN


def test_mamba_model_differs_from_vit_only_on_the_encoder_side(tiny_unetr, tiny_mamba_unetr):
    vit = {name for name, _ in tiny_unetr.named_children()}
    mamba = {name for name, _ in tiny_mamba_unetr.named_children()}
    assert vit - mamba == {"transformer", "decoder3", "decoder6", "decoder9"}
    assert mamba - vit == {"mamba_encoder", "mamba_skip3", "mamba_skip6", "mamba_skip9",
                           "mamba_hidden"}
    vit_sd, mamba_sd = tiny_unetr.state_dict(), tiny_mamba_unetr.state_dict()
    shared = [k for k in vit_sd if k.split(".")[0] in vit & mamba]
    assert shared and all(vit_sd[k].shape == mamba_sd[k].shape for k in shared)
    assert tiny_mamba_unetr.patch_dim == tiny_unetr.patch_dim


def test_mamba_unetr_rejects_shapes_it_cannot_downsample(tiny_mamba_kwargs):
    from models.unetr import UNETR
    with pytest.raises(ValueError, match="divisible by 16"):
        UNETR(**dict(tiny_mamba_kwargs, img_shape=(32, 32, 24)))


def test_mamba_state_dict_round_trips(tiny_mamba_kwargs, tmp_path):
    from models.unetr import UNETR
    torch.manual_seed(0)
    a = UNETR(**tiny_mamba_kwargs).eval()
    torch.manual_seed(1)
    b = UNETR(**tiny_mamba_kwargs).eval()
    torch.save(a.state_dict(), tmp_path / "a.pth")
    b.load_state_dict(torch.load(tmp_path / "a.pth"))
    x = torch.randn(1, 4, *tiny_mamba_kwargs["img_shape"])
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))


# ---------------------------------------------------------------------------
# engine: optimizer, build_model, checkpoints, run snapshots
# ---------------------------------------------------------------------------

def test_mamba_a_log_and_d_are_excluded_from_weight_decay(tiny_mamba_unetr):
    from utils.engine import optimizer_param_groups
    groups = optimizer_param_groups(tiny_mamba_unetr)
    assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0
    no_decay = {id(p) for p in groups[1]["params"]}
    flagged = [p for n, p in tiny_mamba_unetr.named_parameters()
               if n.endswith(".A_log") or n.endswith(".D")]
    # 5 Mamba layers (depths 1,1,1,2) x 3 branches x (A_log, D)
    assert len(flagged) == 30 and {id(p) for p in flagged} == no_decay
    assert sum(len(g["params"]) for g in groups) == len(list(tiny_mamba_unetr.parameters()))


def test_vit_keeps_a_single_param_group_in_model_order(tiny_unetr):
    from utils.engine import optimizer_param_groups
    groups = optimizer_param_groups(tiny_unetr)
    assert len(groups) == 1
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in tiny_unetr.parameters()]


def test_build_model_builds_the_configured_encoder(tiny_mamba_kwargs):
    from config import cfg
    from utils.engine import build_model
    model = build_model(cfg, torch.device("cpu"))
    assert model.encoder_type == "mamba" and hasattr(model, "mamba_encoder")


def test_build_model_refuses_mamba_on_gpu_without_kernels(tiny_mamba_kwargs, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from config import cfg
    from utils.engine import build_model
    monkeypatch.setattr(vm, "_mamba_inner_fn", None)
    with pytest.raises(SystemExit, match="mamba"):
        build_model(cfg, torch.device("cuda"))


def test_resume_refuses_a_checkpoint_from_the_other_encoder(tiny_mamba_unetr, tmp_path,
                                                             monkeypatch):
    from config import cfg
    from utils.checkpoint import load_training_state, save_training_state
    model = tiny_mamba_unetr
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    path = str(tmp_path / "last.pth")
    save_training_state(path, epoch=0, model=model, ema_model=None, optimizer=optimizer,
                        lr_scheduler=scheduler, scaler=None, best_metric=0.0,
                        best_metric_epoch=0, not_improved_epoch=0, cfg=cfg)
    monkeypatch.setattr(cfg.unetr, "encoder", "vit")
    with pytest.raises(SystemExit, match="encoder"):
        load_training_state(path, model=model, ema_model=None, optimizer=optimizer,
                            lr_scheduler=scheduler, scaler=None, cfg=cfg)


def test_evaluate_and_xai_rebuild_the_encoder_a_run_was_trained_with(tmp_path, monkeypatch):
    from config import cfg
    from utils.engine import apply_run_model_config
    monkeypatch.setattr(cfg.unetr, "encoder", "mamba")
    monkeypatch.setattr(cfg.mamba, "dims", [48, 96, 192, 384])

    old_run = tmp_path / "old"
    old_run.mkdir()
    (old_run / "config_snapshot.json").write_text(json.dumps({"unetr": {"embed_dim": 768}}))
    assert apply_run_model_config(cfg, str(old_run)) == "vit"     # pre-switch runs were ViT

    new_run = tmp_path / "new"
    new_run.mkdir()
    (new_run / "config_snapshot.json").write_text(json.dumps(
        {"unetr": {"encoder": "mamba"}, "mamba": {"dims": [8, 16, 32, 64]}}))
    assert apply_run_model_config(cfg, str(new_run)) == "mamba"
    assert list(cfg.mamba.dims) == [8, 16, 32, 64]


def test_config_selects_a_known_encoder():
    from config import cfg
    from models.unetr import ENCODERS
    assert cfg.unetr.encoder in ENCODERS


# ---------------------------------------------------------------------------
# XAI and the training-time attention overlay
# ---------------------------------------------------------------------------

def test_mc_dropout_samples_the_mamba_encoder(tiny_mamba_kwargs):
    from models.unetr import UNETR
    from utils import xai
    kwargs = dict(tiny_mamba_kwargs,
                  mamba_kwargs=dict(tiny_mamba_kwargs["mamba_kwargs"], dropout=0.2))
    torch.manual_seed(0)
    model = UNETR(**kwargs).eval()
    encoder_dropouts = [m for m in model.mamba_encoder.modules()
                        if isinstance(m, torch.nn.Dropout)]
    assert len(encoder_dropouts) == 5 + 4          # one per MambaLayer + one per stage MLP
    assert xai.enable_dropout(model) >= len(encoder_dropouts)

    model.eval()
    for m in encoder_dropouts:                     # ONLY the encoder's dropout active
        m.train()
    x = torch.randn(1, 4, *kwargs["img_shape"])
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x))


def test_training_attention_overlay_works_for_mamba(tiny_mamba_unetr, tiny_mamba_kwargs,
                                                   tmp_path):
    from utils.attention import (extract_attention_map, register_attention_hook,
                                 save_attention_overlay, set_attention_capture)
    shape = tuple(tiny_mamba_kwargs["img_shape"])
    model = tiny_mamba_unetr.eval()
    cache = register_attention_hook(model)

    with torch.no_grad():
        model(torch.randn(1, 4, *shape))
    assert "mamba_tokens" not in cache, "nothing is captured until armed"

    set_attention_capture(cache, True)
    x = torch.randn(1, 4, *shape)
    with torch.no_grad():
        model(x)
    amap = extract_attention_map(cache, model, shape)
    assert amap.shape == shape and np.isfinite(amap).all() and amap.max() > 0

    class _FakeValDataset:
        data = [{"image": ["", "", "/fake/patient1_t1ce.nii", ""]}]

    save_attention_overlay(cache, model, {"image": x}, _FakeValDataset(), 10, shape,
                           str(tmp_path), sample_idx=0)
    assert list(tmp_path.glob("attention_epoch0010_patient1_multislice.png"))

    set_attention_capture(cache, False)
    assert "mamba_tokens" not in cache and "mamba_mixer" not in cache


def test_hidden_attention_rollout_covers_the_input_and_removes_its_hooks(tiny_mamba_unetr,
                                                                        tiny_mamba_kwargs):
    from config import cfg
    from utils import xai
    shape = tuple(tiny_mamba_kwargs["img_shape"])
    model = tiny_mamba_unetr.eval()
    mixers = model.mamba_encoder.last_stage_mixers()
    assert len(mixers) == 2
    before = [len(m._forward_hooks) for m in mixers]

    relevance = xai.attention_rollout(model, torch.randn(1, 4, *shape), cfg)

    assert relevance.shape == shape
    assert relevance.min() >= 0 and relevance.max() <= 1 and relevance.max() > 0
    assert [len(m._forward_hooks) for m in mixers] == before


def test_randomization_cascade_reaches_the_mamba_encoder_and_redraws_ssm_params(
        tiny_mamba_unetr):
    from utils import xai
    model = tiny_mamba_unetr
    order = xai.randomization_order(model)
    assert order[-1] == "mamba_encoder" and "transformer" not in order

    branch = model.mamba_encoder.stages[0][0].mamba.branches["fwd"]
    with torch.no_grad():
        branch.A_log.add_(1.0)
        branch.D.add_(1.0)
        branch.dt_proj.bias.fill_(5.0)
    xai._reinitialize(model.mamba_encoder)

    expected_A = torch.log(torch.arange(1, branch.d_state + 1.0).repeat(branch.d_inner, 1))
    torch.testing.assert_close(branch.A_log, expected_A)
    assert torch.all(branch.D == 1)
    dt = F.softplus(branch.dt_proj.bias)
    assert dt.min() >= 1e-3 - 1e-6 and dt.max() <= 0.1 + 1e-6, \
        "the branch must re-apply Mamba's dt init AFTER dt_proj resets itself"


def test_full_xai_suite_runs_on_the_mamba_model(tiny_mamba_unetr, tiny_mamba_kwargs,
                                                tmp_path, monkeypatch):
    from config import cfg
    from utils import xai
    shape = tuple(tiny_mamba_kwargs["img_shape"])
    monkeypatch.setattr(cfg.xai, "sample_indices", [0])
    monkeypatch.setattr(cfg.xai, "mc_passes", 2)
    monkeypatch.setattr(cfg.xai, "deletion_fractions", [0.0, 0.1, 0.5])
    monkeypatch.setattr(cfg.xai, "cam_layers", ["decoder9_upsampler", "decoder0_header.1"])

    torch.manual_seed(0)
    image = torch.randn(4, *shape)
    label = torch.zeros(3, *shape)
    label[:, 8:20, 8:20, 4:12] = 1.0

    class _Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, _idx):
            return {"image": image, "label": label}

    model = tiny_mamba_unetr.eval()
    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    out_dir = tmp_path / "xai"

    summary = xai.run_xai_suite(
        model=model, test_ds=_Dataset(),
        test_loader=[{"image": image[None], "label": label[None]}],
        checkpoint_path=str(checkpoint), out_dir=str(out_dir), device=torch.device("cpu"),
        cfg=cfg, inferer=lambda m, x: m(x), components=xai.COMPONENTS,
    )

    failed = {k: v["error"] for k, v in summary.items() if isinstance(v, dict) and "error" in v}
    assert not failed, failed
    for name in xai.COMPONENTS:
        assert (out_dir / f"{name}.json").exists(), name
    assert summary["faithful"]["sanity_check"]["cascade"][-1]["randomized_through"] == \
        "mamba_encoder"
    assert list(out_dir.glob("rollout_s0.png"))


# ---------------------------------------------------------------------------
# full size, on the GPU (pytest -m slow, from the `mamba` env)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_kernels
def test_real_config_mamba_unetr_trains_one_step_under_bf16():
    from config import cfg
    from models.unetr import UNETR
    torch.manual_seed(0)
    model = UNETR(img_shape=cfg.unetr.img_shape, input_dim=cfg.unetr.input_dim,
                  output_dim=cfg.unetr.output_dim, embed_dim=cfg.unetr.embed_dim,
                  patch_size=cfg.unetr.patch_size, num_heads=cfg.unetr.num_heads,
                  dropout=cfg.unetr.dropout, encoder="mamba",
                  mamba_kwargs=dict(cfg.mamba)).cuda().train()
    x = torch.randn(2, cfg.unetr.input_dim, *cfg.unetr.img_shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, aux_z6, aux_z3 = model(x)
        loss = out.float().mean() + aux_z6.float().mean() + aux_z3.float().mean()
    loss.backward()
    assert out.shape == (2, cfg.unetr.output_dim, *cfg.unetr.img_shape)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
