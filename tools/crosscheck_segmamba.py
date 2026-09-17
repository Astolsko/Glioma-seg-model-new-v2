"""Numerical cross-check of blocks/VisionMamba.py against the OFFICIAL SegMamba code.

    conda activate mamba
    python tools/crosscheck_segmamba.py

Downloads SegMamba's model_segmamba/segmamba.py and its forked
mamba/mamba_ssm/modules/mamba_simple.py (bimamba_type="v3") at a pinned commit,
builds their MambaEncoder next to ours, copies our weights into theirs by name
(load_state_dict(strict=True), so any parameter either side has and the other
lacks fails the check), then compares, on a full-size 128x128x96 volume in
fp32 on the GPU:
  * every MambaLayer in isolation: the official layer is fed exactly the input
    ours received, so a structural difference in any one layer cannot hide
    behind the others;
  * all four stage outputs end to end.
TF32 is switched off for the comparison. cuDNN convolutions default to TF32 on
Ampere, which keeps 10 mantissa bits, so a 1e-7 rounding difference between two
equivalent formulations of the Mamba ops becomes ~1e-3 after the next conv and
would drown the signal this check exists to measure.

Two shims are needed to import the official files against upstream mamba_ssm:
  * their fork's `mamba_inner_fn_no_out_proj` is not in upstream. It is
    mamba_inner_fn without the output projection; here it is rebuilt from
    upstream's causal_conv1d_fn + selective_scan_fn, exactly as mamba_inner_ref
    computes it minus the final F.linear.
  * segmamba.py does `from mamba_ssm import Mamba` expecting THEIR forked class;
    that line is pointed at the downloaded fork instead.
"""
import os
import sys
import types
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEGMAMBA_COMMIT = "cff35970e0c542ad940b5701267f1ac888298b06"
RAW = f"https://raw.githubusercontent.com/ge-xing/SegMamba/{SEGMAMBA_COMMIT}"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "segmamba_reference", SEGMAMBA_COMMIT)


def _fetch(rel_path):
    local = os.path.join(CACHE, rel_path.replace("/", "__"))
    if not os.path.exists(local):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(f"{RAW}/{rel_path}", local)
    with open(local) as f:
        return f.read()


def load_official_encoder_class():
    import torch.nn.functional as F
    from einops import rearrange
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    def mamba_inner_fn_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                                   delta_proj_weight, A, B=None, C=None, D=None,
                                   delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                                   delta_softplus=True):
        L = xz.shape[-1]
        rank, n = delta_proj_weight.shape[1], A.shape[-1]
        x, z = xz.chunk(2, dim=1)
        x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias,
                             activation="silu")
        x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :rank].t(), "d (b l) -> b d l", l=L)
        B = rearrange(x_dbl[:, rank:rank + n], "(b l) n -> b n l", l=L).contiguous()
        C = rearrange(x_dbl[:, -n:], "(b l) n -> b n l", l=L).contiguous()
        return selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias,
                                 delta_softplus=delta_softplus)

    shim = types.ModuleType("_segmamba_shim")
    shim.mamba_inner_fn_no_out_proj = mamba_inner_fn_no_out_proj
    sys.modules["_segmamba_shim"] = shim

    fork_src = _fetch("mamba/mamba_ssm/modules/mamba_simple.py")
    old = ("from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, "
           "mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj")
    assert fork_src.count(old) == 1, "the fork's import line moved; update the shim"
    fork_src = fork_src.replace(
        old, "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, "
             "mamba_inner_fn\n    from _segmamba_shim import mamba_inner_fn_no_out_proj\n"
             "    bimamba_inner_fn = None")
    fork = {"__name__": "segmamba_mamba_simple"}
    exec(compile(fork_src, "segmamba/mamba_simple.py", "exec"), fork)

    model_src = _fetch("model_segmamba/segmamba.py")
    assert model_src.count("from mamba_ssm import Mamba\n") == 1
    model_src = model_src.replace("from mamba_ssm import Mamba\n", "")
    official = {"__name__": "segmamba_model", "Mamba": fork["Mamba"]}
    exec(compile(model_src, "segmamba/segmamba.py", "exec"), official)
    return official["MambaEncoder"]


BRANCH_SUFFIX = {"fwd": "", "bwd": "_b", "slice": "_s"}


def ours_to_official_key(key):
    """blocks.VisionMamba parameter names -> SegMamba's."""
    marker = ".mamba.branches."
    if marker not in key:
        return key                    # stem/downsample/gscs/norm/mlps/in_proj/out_proj
    head, tail = key.split(marker)
    direction, param = tail.split(".", 1)
    suffix = BRANCH_SUFFIX[direction]
    if param == "A_log":
        name = "A_log" if not suffix else f"A{suffix}_log"
    elif param == "D":
        name = f"D{suffix}"
    else:
        module, attr = param.split(".", 1)
        name = f"{module}{suffix}.{attr}"
    return f"{head}.mamba.{name}"


def main():
    import torch
    from config import cfg
    from blocks.VisionMamba import VisionMambaEncoder, mamba_kernels_available

    if not (torch.cuda.is_available() and mamba_kernels_available()):
        raise SystemExit("needs a GPU and the mamba_ssm kernels: conda activate mamba")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    OfficialEncoder = load_official_encoder_class()
    torch.manual_seed(0)
    ours = VisionMambaEncoder(in_chans=cfg.unetr.input_dim, dims=cfg.mamba.dims,
                              depths=cfg.mamba.depths, d_state=cfg.mamba.d_state,
                              d_conv=cfg.mamba.d_conv, expand=cfg.mamba.expand,
                              dropout=0.0).cuda().eval()
    theirs = OfficialEncoder(in_chans=cfg.unetr.input_dim, depths=list(cfg.mamba.depths),
                             dims=list(cfg.mamba.dims)).cuda().eval()

    mapped = {ours_to_official_key(k): v for k, v in ours.state_dict().items()}
    result = theirs.load_state_dict(mapped, strict=True)
    n_ours = sum(p.numel() for p in ours.parameters())
    n_theirs = sum(p.numel() for p in theirs.parameters())
    print(f"strict load OK ({result}) | params ours={n_ours:,} official={n_theirs:,}")
    assert n_ours == n_theirs

    x = torch.randn(1, cfg.unetr.input_dim, *cfg.unetr.img_shape, device="cuda")

    # Layer by layer: capture what each of OUR MambaLayers receives, feed the
    # same tensor to the matching official layer.
    captured = {}
    handles = []
    for i, stage in enumerate(ours.stages):
        for j, layer in enumerate(stage):
            def _hook(_m, inputs, output, key=(i, j)):
                captured[key] = (inputs[0].detach(), output.detach())
            handles.append(layer.register_forward_hook(_hook))
    with torch.no_grad():
        out_ours = ours(x)
    for h in handles:
        h.remove()
    layer_worst = 0.0
    with torch.no_grad():
        for (i, j), (inp, out) in sorted(captured.items()):
            ref = theirs.stages[i][j](inp)
            rel = ((out - ref).norm() / ref.norm()).item()
            layer_worst = max(layer_worst, rel)
            print(f"stage {i + 1} MambaLayer {j}: grid {tuple(inp.shape[2:])} "
                  f"(nslices {inp.shape[2]}) | relative L2 difference {rel:.2e}")
    assert layer_worst < 1e-5, "A MAMBA LAYER DOES NOT MATCH THE OFFICIAL SEGMAMBA LAYER"

    with torch.no_grad():
        out_theirs = theirs(x)
    worst = 0.0
    for i, (a, b) in enumerate(zip(out_ours, out_theirs)):
        rel = ((a - b).norm() / b.norm()).item()
        worst = max(worst, rel)
        print(f"stage {i + 1}: shape {tuple(a.shape)} | relative L2 difference {rel:.2e}")
    assert worst < 1e-4, "OUR ENCODER DOES NOT MATCH THE OFFICIAL SEGMAMBA ENCODER"
    worst = max(worst, layer_worst)
    print(f"CROSS-CHECK PASSED: identical to SegMamba@{SEGMAMBA_COMMIT[:7]} "
          f"(worst relative difference {worst:.1e})")


if __name__ == "__main__":
    main()
