"""
DepthFlow CUDA Renderer — GPU-accelerated parallax rendering without OpenGL.

This module implements the DepthFlow rendering pipeline entirely on CUDA via
PyTorch, matching the GLSL output pixel-for-pixel.  It can be used:

  1. By the DepthFlow CLI  (``depthflow scene --backend cuda …``)
  2. By ComfyUI nodes that call ``CudaDepthFlowRenderer`` directly — this
     avoids subprocess overhead *and* the OpenGL/EGL dependency entirely.

Requirements: ``torch`` built with CUDA support.
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ────────────────────────────────────────────────────────────────────────────
# Native CUDA kernel — compiled on first use via load_inline
# ────────────────────────────────────────────────────────────────────────────

_NATIVE_MODULE = None   # lazy singleton
_NATIVE_TRIED  = False

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ── helpers ─────────────────────────────────────────────────────────── */

__device__ __forceinline__ float tri_wave(float x, float period) {
    float t = 2.0f * x / period - 0.5f;
    t = fmodf(t, 2.0f);
    if (t < 0.0f) t += 2.0f;
    return 2.0f * fabsf(t - 1.0f) - 1.0f;
}

__device__ __forceinline__ void gluv2grid(
    float px, float py, bool mirror, float wa, float sx,
    float &gx, float &gy
) {
    if (mirror) {
        gx = wa * tri_wave(px, 4.0f * wa) * sx;
        gy = -tri_wave(py, 4.0f);
    } else {
        gx = px * sx;
        gy = -py;
    }
}

__device__ __forceinline__ float bilerp(
    const float* __restrict__ d, int w, int h, float gx, float gy
) {
    float fx = (gx + 1.0f) * w * 0.5f - 0.5f;
    float fy = (gy + 1.0f) * h * 0.5f - 0.5f;
    fx = fmaxf(0.0f, fminf((float)(w - 1), fx));
    fy = fmaxf(0.0f, fminf((float)(h - 1), fy));
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float wx = fx - (float)x0, wy = fy - (float)y0;
    return (1-wx)*(1-wy) * __ldg(&d[y0*w+x0])
         +    wx *(1-wy) * __ldg(&d[y0*w+x1])
         + (1-wx)*   wy  * __ldg(&d[y1*w+x0])
         +    wx *   wy  * __ldg(&d[y1*w+x1]);
}

/* ── forward march (matches GLSL Stage 0 exactly) ────────────────────── */

__global__ void forward_march_k(
    const float* __restrict__ depth,
    const float* __restrict__ ox, const float* __restrict__ oy,
    const float* __restrict__ ix, const float* __restrict__ iy,
    float* __restrict__ walk_out, int* __restrict__ hit_out,
    float oz, int rw, int rh, int iw, int ih,
    float dh, float di, bool mirror, float wa,
    float probe, float safe, float sx
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rw || y >= rh) return;
    int idx = y * rw + x;

    float o_x = __ldg(&ox[idx]), o_y = __ldg(&oy[idx]);
    float i_x = __ldg(&ix[idx]), i_y = __ldg(&iy[idx]);

    // Match GLSL: walk starts at 0, check walk>1.0 BEFORE increment
    float w = 0.0f;
    for (int it = 0; it < 1000; it++) {
        if (w > 1.0f) break;
        w += probe;

        float mt = safe + (1.0f - safe) * w;
        float c1 = 1.0f - mt;
        float px = o_x * c1 + i_x * mt;
        float py = o_y * c1 + i_y * mt;
        float ceil_v = 1.0f - (oz * c1 + mt);

        float gx, gy;
        gluv2grid(px, py, mirror, wa, sx, gx, gy);
        float dv = bilerp(depth, iw, ih, gx, gy);
        float surf = dh * (dv * (1.0f - di) + (1.0f - dv) * di);

        if (surf > ceil_v) { walk_out[idx] = w; hit_out[idx] = 1; return; }
    }
    walk_out[idx] = 1.0f;
    hit_out[idx] = 0;
}

/* ── backward linear refinement (matches GLSL Stage 1 exactly) ────────── */

__global__ void backward_refine_k(
    const float* __restrict__ depth,
    const float* __restrict__ ox, const float* __restrict__ oy,
    const float* __restrict__ ix, const float* __restrict__ iy,
    const float* __restrict__ walk_in, const int* __restrict__ hit_in,
    float* __restrict__ walk_out,
    float oz, int npx, int iw, int ih,
    float dh, float di, bool mirror, float wa,
    float safe, float sx, float quality_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npx) return;
    if (!hit_in[idx]) { walk_out[idx] = walk_in[idx]; return; }

    float o_x = __ldg(&ox[idx]), o_y = __ldg(&oy[idx]);
    float i_x = __ldg(&ix[idx]), i_y = __ldg(&iy[idx]);
    float w = walk_in[idx];
    float last_value = 0.0f;

    // Match GLSL Stage 1: walk backwards, stop when outside surface
    for (int it = 0; it < 1000; it++) {
        w -= quality_step;
        // GLSL does NOT clamp walk < 0. Match this.
        // In practice we clamp at a safe minimum to avoid garbage.
        if (w < -0.1f) { w = 0.0f; break; }

        float mt = safe + (1.0f - safe) * w;
        float c1 = 1.0f - mt;
        float ceil_v = 1.0f - (oz * c1 + mt);

        float gx, gy;
        gluv2grid(o_x * c1 + i_x * mt, o_y * c1 + i_y * mt,
                  mirror, wa, sx, gx, gy);
        float dv = bilerp(depth, iw, ih, gx, gy);
        float surf = dh * (dv * (1.0f - di) + (1.0f - dv) * di);

        // GLSL: ceiling < surface → inside (continue)
        //       else (BACKWARD) → break (found outside edge)
        if (ceil_v >= surf) break;  // outside → this is our answer
    }
    walk_out[idx] = fmaxf(w, 0.0f);
}

/* ── walk → gluv recovery ────────────────────────────────────────────── */

__global__ void walk2gluv_k(
    const float* __restrict__ ox, const float* __restrict__ oy,
    const float* __restrict__ ix, const float* __restrict__ iy,
    const float* __restrict__ walk, const int* __restrict__ hit,
    const float* __restrict__ cgx, const float* __restrict__ cgy,
    float* __restrict__ rgx, float* __restrict__ rgy,
    float safe, int npx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npx) return;
    // For hit pixels, walk is the refined boundary; for no-hit, walk=1.0
    // which gives mix_t=1.0 → point = intersect (correct GLSL behavior)
    float mt = safe + (1.0f - safe) * walk[idx];
    float c1 = 1.0f - mt;
    rgx[idx] = ox[idx] * c1 + ix[idx] * mt;
    rgy[idx] = oy[idx] * c1 + iy[idx] * mt;
}

/* ── Python bindings ──────────────────────────────────────────────────── */

std::vector<torch::Tensor> native_forward_march(
    torch::Tensor depth, torch::Tensor orig_x, torch::Tensor orig_y,
    torch::Tensor int_x, torch::Tensor int_y,
    double oz, int rw, int rh, int iw, int ih,
    double dh, double di, bool mirror, double wa,
    double probe, double safe, double sx
) {
    auto opt = orig_x.options();
    auto walk = torch::ones({rh, rw}, opt);
    auto hit  = torch::zeros({rh, rw}, opt.dtype(torch::kInt32));
    dim3 blk(16, 16);
    dim3 grd((rw+15)/16, (rh+15)/16);
    forward_march_k<<<grd, blk>>>(
        depth.data_ptr<float>(),
        orig_x.data_ptr<float>(), orig_y.data_ptr<float>(),
        int_x.data_ptr<float>(), int_y.data_ptr<float>(),
        walk.data_ptr<float>(), hit.data_ptr<int>(),
        (float)oz, rw, rh, iw, ih,
        (float)dh, (float)di, mirror, (float)wa,
        (float)probe, (float)safe, (float)sx);
    return {walk, hit};
}

std::vector<torch::Tensor> native_bisect(
    torch::Tensor depth, torch::Tensor orig_x, torch::Tensor orig_y,
    torch::Tensor int_x, torch::Tensor int_y,
    torch::Tensor walk_in, torch::Tensor hit_in,
    double oz, int iw, int ih,
    double dh, double di, bool mirror, double wa,
    double safe, double sx, double quality_step
) {
    int npx = orig_x.numel();
    auto wout = walk_in.clone();
    int t = 256, b = (npx + t - 1) / t;
    backward_refine_k<<<b, t>>>(
        depth.data_ptr<float>(),
        orig_x.data_ptr<float>(), orig_y.data_ptr<float>(),
        int_x.data_ptr<float>(), int_y.data_ptr<float>(),
        walk_in.data_ptr<float>(), hit_in.data_ptr<int>(),
        wout.data_ptr<float>(),
        (float)oz, npx, iw, ih,
        (float)dh, (float)di, mirror, (float)wa,
        (float)safe, (float)sx, (float)quality_step);
    return {wout};
}

torch::Tensor native_walk2gluv(
    torch::Tensor orig_x, torch::Tensor orig_y,
    torch::Tensor int_x, torch::Tensor int_y,
    torch::Tensor walk, torch::Tensor hit,
    torch::Tensor cam_gx, torch::Tensor cam_gy,
    double safe
) {
    int npx = orig_x.numel();
    auto rgx = torch::empty_like(orig_x);
    auto rgy = torch::empty_like(orig_y);
    int t = 256, b = (npx + t - 1) / t;
    walk2gluv_k<<<b, t>>>(
        orig_x.data_ptr<float>(), orig_y.data_ptr<float>(),
        int_x.data_ptr<float>(), int_y.data_ptr<float>(),
        walk.data_ptr<float>(), hit.data_ptr<int>(),
        cam_gx.data_ptr<float>(), cam_gy.data_ptr<float>(),
        rgx.data_ptr<float>(), rgy.data_ptr<float>(),
        (float)safe, npx);
    return torch::stack({rgx, rgy});
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> native_forward_march(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    double, int, int, int, int,
    double, double, bool, double,
    double, double, double);
std::vector<torch::Tensor> native_bisect(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    double, int, int, double, double, bool, double,
    double, double, double);
torch::Tensor native_walk2gluv(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
"""


def _get_native_module():
    """Lazily compile (and cache) the native CUDA extension."""
    global _NATIVE_MODULE, _NATIVE_TRIED
    if _NATIVE_TRIED:
        return _NATIVE_MODULE
    _NATIVE_TRIED = True
    if not _HAS_TORCH or not torch.cuda.is_available():
        return None
    try:
        # Ensure ninja is on PATH (pip-installed ninja lives in venv/bin)
        import ninja, os
        os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
        from torch.utils.cpp_extension import load_inline
        _NATIVE_MODULE = load_inline(
            name="depthflow_native_v3",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=[
                "native_forward_march",
                "native_bisect",
                "native_walk2gluv",
            ],
            verbose=False,
            with_cuda=True,
        )
        print("[DepthFlow CUDA] Native kernel compiled OK")
    except Exception as exc:
        print(f"[DepthFlow CUDA] Native compile skipped: {exc}")
        _NATIVE_MODULE = None
    return _NATIVE_MODULE


# ────────────────────────────────────────────────────────────────────────────
# Animation helpers — pure Python, no DepthFlow/ShaderFlow dependency
# ────────────────────────────────────────────────────────────────────────────

def _triangle_wave(x: float, period: float) -> float:
    """Matches GLSL ``triangle_wave(x, period)``."""
    return 2.0 * abs(((2.0 * x / period - 0.5) % 2.0) - 1.0) - 1.0


class DepthFlowState:
    """Mirror of ``DepthState`` + post-processing — pure Python."""

    __slots__ = (
        "height", "steady", "focus", "zoom", "isometric", "dolly", "invert",
        "mirror", "offset_x", "offset_y", "center_x", "center_y",
        "origin_x", "origin_y",
        "vig_enable", "vig_intensity", "vig_decay",
        "lens_enable", "lens_intensity", "lens_decay", "lens_quality",
        "blur_enable", "blur_intensity", "blur_start", "blur_end",
        "blur_exponent", "blur_quality", "blur_directions",
        "color_saturation", "color_contrast", "color_brightness",
        "color_gamma", "color_grayscale", "color_sepia",
    )

    def __init__(self) -> None:
        self.height: float = 0.20
        self.steady: float = 0.0
        self.focus: float = 0.0
        self.zoom: float = 1.0
        self.isometric: float = 0.0
        self.dolly: float = 0.0
        self.invert: float = 0.0
        self.mirror: bool = True
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.vig_enable: bool = False
        self.vig_intensity: float = 0.2
        self.vig_decay: float = 20.0
        self.lens_enable: bool = False
        self.lens_intensity: float = 0.1
        self.lens_decay: float = 0.4
        self.lens_quality: int = 30
        self.blur_enable: bool = False
        self.blur_intensity: float = 1.0
        self.blur_start: float = 0.6
        self.blur_end: float = 1.0
        self.blur_exponent: float = 2.0
        self.blur_quality: int = 4
        self.blur_directions: int = 16
        self.color_saturation: float = 1.0
        self.color_contrast: float = 1.0
        self.color_brightness: float = 1.0
        self.color_gamma: float = 1.0
        self.color_grayscale: float = 0.0
        self.color_sepia: float = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Animation presets  (matches DepthFlow Animation.*)
# ────────────────────────────────────────────────────────────────────────────

def _compute_sine(tau: float, cycle: float, amplitude: float, phase: float,
                  cycles: float = 1.0, bias: float = 0.0) -> float:
    return amplitude * math.sin(cycle * cycles + phase * math.tau) + bias


def _compute_cosine(tau: float, cycle: float, amplitude: float, phase: float,
                    cycles: float = 1.0, bias: float = 0.0) -> float:
    return amplitude * math.cos(cycle * cycles + phase * math.tau) + bias


def _compute_triangle(tau: float, amplitude: float, phase: float,
                      cycles: float = 1.0, bias: float = 0.0) -> float:
    t = (tau * cycles + phase + 0.25) % 1.0
    return amplitude * (1.0 - 4.0 * abs(t - 0.5)) + bias


def compute_animation_state(
    camera_movement: str,
    tau: float,
    *,
    intensity: float = 1.0,
    smooth: bool = True,
    loop: bool = True,
    reverse: bool = False,
    phase: float = 0.0,
    steady_depth: float = 0.3,
    isometric: float = 0.6,
) -> DepthFlowState:
    """Computes a single frame's ``DepthFlowState`` from animation params.

    ``tau`` goes from 0→1 over the video duration.
    """
    state = DepthFlowState()
    cycle = 2.0 * math.pi * tau

    if reverse:
        cycle = 2.0 * math.pi - cycle
        tau = 1.0 - tau

    move = camera_movement.lower().strip()

    if move == "static":
        state.height = 0.2
        return state

    if move == "vertical":
        state.isometric = isometric
        state.steady = steady_depth
        if loop:
            state.offset_y = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, 0.8 * intensity, phase, cycles=1.0,
            ) if smooth else _compute_triangle(
                tau, 0.8 * intensity, phase, cycles=1.0,
            )
        else:
            state.offset_y = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, intensity, -0.25, cycles=0.5,
            ) if smooth else _compute_triangle(
                tau, intensity, -0.25, cycles=0.5,
            )

    elif move == "horizontal":
        state.isometric = isometric
        state.steady = steady_depth
        if loop:
            state.offset_x = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, 0.8 * intensity, phase, cycles=1.0,
            ) if smooth else _compute_triangle(
                tau, 0.8 * intensity, phase, cycles=1.0,
            )
        else:
            state.offset_x = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, intensity, -0.25, cycles=0.5,
            ) if smooth else _compute_triangle(
                tau, intensity, -0.25, cycles=0.5,
            )

    elif move == "zoom":
        state.isometric = isometric
        if loop:
            state.height = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, intensity / 2.0, phase, cycles=1.0, bias=intensity / 2.0,
            ) if smooth else _compute_triangle(
                tau, intensity / 2.0, phase, cycles=1.0, bias=intensity / 2.0,
            )
        else:
            state.height = (_compute_sine if smooth else _compute_triangle)(
                tau, cycle, 2.0 * intensity, 0.0, cycles=0.25,
            ) if smooth else _compute_triangle(
                tau, 2.0 * intensity, 0.0, cycles=0.25,
            )

    elif move == "circle":
        state.isometric = isometric
        state.steady = steady_depth
        state.offset_x = _compute_sine(
            tau, cycle, 0.5 * intensity, phase + 0.25,
        )
        state.offset_y = _compute_sine(
            tau, cycle, 0.5 * intensity, phase,
        )

    elif move == "dolly":
        state.height = intensity / 3.0
        state.steady = steady_depth
        state.focus = steady_depth
        if loop:
            p = 0.75 if reverse else 0.25
            c = 1.0
        else:
            p = -0.75 if reverse else 0.25
            c = 0.5

        # Original Dolly passes reverse=(not self.reverse) to its inner Sine
        dolly_reverse = not reverse
        if dolly_reverse:
            inner_cycle = 2.0 * math.pi - cycle
            inner_tau = 1.0 - tau
        else:
            inner_cycle = cycle
            inner_tau = tau

        val = (_compute_sine if smooth else _compute_triangle)(
            inner_tau, inner_cycle, intensity / 2.0, phase + p, cycles=c, bias=intensity / 2.0,
        ) if smooth else _compute_triangle(
            inner_tau, intensity / 2.0, phase + p, cycles=c, bias=intensity / 2.0,
        )
        state.isometric = val

    elif move == "orbital":
        state.steady = steady_depth
        state.focus = steady_depth
        state.zoom = 0.98
        state.isometric = _compute_cosine(
            tau, cycle, intensity / 4.0, 0.0, bias=intensity / 2.0 + 0.5,
        )
        state.offset_x = _compute_sine(
            tau, cycle, intensity / 4.0, 0.0,
        )

    return state



# ────────────────────────────────────────────────────────────────────────────
# GLSL-matching helpers (HSV, smoothstep)
# ────────────────────────────────────────────────────────────────────────────

def _rgb_to_hsv(r: "torch.Tensor", g: "torch.Tensor", b: "torch.Tensor"
                ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Vectorised RGB→HSV matching ``rgb2hsv`` in shaderflow.glsl."""
    cmax = torch.max(torch.max(r, g), b)
    cmin = torch.min(torch.min(r, g), b)
    delta = cmax - cmin

    # Hue
    h = torch.zeros_like(r)
    nonzero = delta > 1e-8
    mask_r = nonzero & (cmax == r)
    mask_g = nonzero & (cmax == g) & ~mask_r
    mask_b = nonzero & ~mask_r & ~mask_g
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]).fmod(6.0)
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
    h = h * (math.pi / 3.0)  # radians

    # Saturation
    s = torch.where(cmax > 1e-8, delta / cmax, torch.zeros_like(cmax))
    return h, s, cmax  # (H, S, V)


def _hsv_to_rgb(h: "torch.Tensor", s: "torch.Tensor", v: "torch.Tensor"
                ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Vectorised HSV→RGB matching ``hsv2rgb`` in shaderflow.glsl."""
    TAU = 2.0 * math.pi
    h = h.fmod(TAU)
    h = torch.where(h < 0, h + TAU, h)  # ensure [0, 2π)
    c = v * s
    x = c * (1.0 - ((h / (math.pi / 3.0)).fmod(2.0) - 1.0).abs())
    m = v - c

    sector = (h / TAU * 6.0).floor().long().clamp(0, 5)
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    for idx, (rv, gv, bv) in enumerate([
        ("c", "x", "z"), ("x", "c", "z"), ("z", "c", "x"),
        ("z", "x", "c"), ("x", "z", "c"), ("c", "z", "x"),
    ]):
        mask = sector == idx
        for ch, key in [(r, rv), (g, gv), (b, bv)]:
            if key == "c":
                ch[mask] = c[mask]
            elif key == "x":
                ch[mask] = x[mask]
            # "z" → 0 (already initialised)
    return r + m, g + m, b + m


def _smoothstep_t(edge0: float, edge1: float, x: "torch.Tensor") -> "torch.Tensor":
    """GLSL ``smoothstep`` for tensors."""
    t = ((x - edge0) / max(edge1 - edge0, 1e-8)).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


# ────────────────────────────────────────────────────────────────────────────
# CUDA Renderer  (PyTorch, no OpenGL)
# ────────────────────────────────────────────────────────────────────────────

def _check_cuda() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def _sample_texture(tex: torch.Tensor, grid: torch.Tensor,
                    mirror: bool) -> torch.Tensor:
    """Sample ``tex`` NCHW at ``grid`` NHW2 with border-clamp or reflect."""
    mode = "border"  # clamp-to-edge
    return F.grid_sample(tex, grid, mode="bilinear",
                         padding_mode=mode, align_corners=False)


class CudaDepthFlowRenderer:
    """Render DepthFlow parallax frames entirely on CUDA (PyTorch).

    Parameters
    ----------
    image : array-like  (H, W, 3) float32 [0, 1]
    depth : array-like  (H, W)    float32 [0, 1]
    device : str
    """

    def __init__(
        self,
        image: Any,
        depth: Any,
        device: str = "cuda",
        depth_post_process: bool = True,
    ) -> None:
        if not _check_cuda():
            raise RuntimeError("CUDA not available — cannot use CudaDepthFlowRenderer")
        self.device = torch.device(device)

        # Convert to torch tensors on GPU — NCHW layout for grid_sample
        img = self._to_tensor(image)  # (H, W, 3)
        dep = self._to_tensor(depth)  # (H, W) or (H, W, 1)
        if dep.ndim == 3:
            dep = dep[..., 0]

        self.img_h, self.img_w = img.shape[0], img.shape[1]
        # (1, 3, H, W) and (1, 1, H, W)
        self.image_gpu = img.permute(2, 0, 1).unsqueeze(0).to(self.device).contiguous()
        self.depth_gpu = dep.unsqueeze(0).unsqueeze(0).to(self.device).contiguous()

        # Match DepthFlow DA2 depth post-processing:
        # 1) Gaussian blur σ=0.6 — smooth high-frequency noise
        # 2) 5×5 max-pool (dilation) — thicken foreground edges to prevent
        #    background pixels from "peeking through" at silhouette boundaries
        if depth_post_process:
            self.depth_gpu = self._post_process_depth(self.depth_gpu)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _post_process_depth(depth_gpu: "torch.Tensor") -> "torch.Tensor":
        """Match DepthFlow DA2 _post(): Gaussian(σ=0.6) + MaxPool(5)."""
        # Gaussian blur kernel (7×7, σ=0.6)
        sigma = 0.6
        ks = 7  # kernel size (must be odd)
        half = ks // 2
        coords = torch.arange(ks, dtype=torch.float32, device=depth_gpu.device) - half
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # (ks, ks)
        kernel = kernel.unsqueeze(0).unsqueeze(0)      # (1, 1, ks, ks)
        blurred = F.conv2d(
            F.pad(depth_gpu, (half, half, half, half), mode="replicate"),
            kernel,
        )
        # 5×5 max-pool (foreground edge dilation, stride=1)
        dilated = F.max_pool2d(blurred, kernel_size=5, stride=1, padding=2)
        return dilated

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    # ------------------------------------------------------------------ coords

    @staticmethod
    def _make_gluv_grid(
        render_w: int, render_h: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pixel-center gluv in OpenGL convention (y-up)."""
        xs = (2.0 * (torch.arange(render_w, device=device, dtype=torch.float32) + 0.5) / render_w - 1.0)
        ys = 1.0 - 2.0 * (torch.arange(render_h, device=device, dtype=torch.float32) + 0.5) / render_h
        gluv_y, gluv_x = torch.meshgrid(ys, xs, indexing="ij")
        return gluv_x, gluv_y  # each (H, W)

    def _gluv_to_grid(
        self, gluv_x: torch.Tensor, gluv_y: torch.Tensor,
        mirror: bool, want_aspect: float,
    ) -> torch.Tensor:
        """Convert scene-gluv → ``grid_sample`` grid  (N=1, H, W, 2)."""
        if mirror:
            gluv_x = want_aspect * _triangle_wave_t(gluv_x, 4.0 * want_aspect)
            gluv_y = _triangle_wave_t(gluv_y, 4.0)

        # gtexture: scale = (tex_h/tex_w, 1); stuv = (gluv*scale +1)/2
        # grid_sample coord = 2*stuv - 1 = gluv * scale
        scale_x = float(self.img_h) / float(self.img_w)
        grid_x = gluv_x * scale_x   # horizontal
        grid_y = -gluv_y             # flip Y (OpenGL → PyTorch)
        # stack → (H, W, 2), add batch
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    # -------------------------------------------------------- optimised march

    def _ray_march_v2(
        self,
        orig_x: torch.Tensor, orig_y: torch.Tensor, orig_z: torch.Tensor,
        int_x: torch.Tensor, int_y: torch.Tensor, int_z: torch.Tensor,
        cam_gluv_x: torch.Tensor, cam_gluv_y: torch.Tensor,
        df_height: float, df_invert: float,
        mirror: bool, want_aspect: float,
        quality_norm: float,
        render_h: int, render_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ray-march dispatcher — uses native CUDA kernel when available,
        otherwise falls back to a zero-sync PyTorch loop.

        Returns ``(result_gluv_x, result_gluv_y, value)`` where
        *value* has shape ``(1, 1, H, W)`` (depth at intersection).
        """
        dev = self.device
        probe_step = 1.0 / (50.0 + 70.0 * quality_norm)
        safe = 1.0 - df_height
        scale_x = float(self.img_h) / float(self.img_w)
        orig_z_val = float(orig_z.view(-1)[0].item())

        # ── Native CUDA kernel path (fastest) ──────────────────────────
        native = _get_native_module()
        if native is not None:
            depth_flat = self.depth_gpu.view(self.img_h, self.img_w).contiguous()
            ox = orig_x.contiguous()
            oy = orig_y.contiguous()
            ix = int_x.contiguous()
            iy = int_y.contiguous()

            walk, hit = native.native_forward_march(
                depth_flat, ox, oy, ix, iy,
                orig_z_val, render_w, render_h,
                self.img_w, self.img_h,
                df_height, df_invert, mirror, want_aspect,
                probe_step, safe, scale_x,
            )

            # Backward linear refinement (matches GLSL Stage 1)
            quality_step = 1.0 / (200.0 + 1800.0 * quality_norm)
            wout_list = native.native_bisect(
                depth_flat, ox, oy, ix, iy,
                walk, hit,
                orig_z_val, self.img_w, self.img_h,
                df_height, df_invert, mirror, want_aspect,
                safe, scale_x, quality_step,
            )
            wlo = wout_list[0]

            # Recover gluv from walk values
            gluv_stack = native.native_walk2gluv(
                ox, oy, ix, iy,
                wlo, hit,
                cam_gluv_x.contiguous(), cam_gluv_y.contiguous(),
                safe,
            )
            result_gluv_x = gluv_stack[0]
            result_gluv_y = gluv_stack[1]

            # Depth at result for post-processing → (1, 1, H, W)
            grid_final = self._gluv_to_grid(
                result_gluv_x, result_gluv_y, mirror, want_aspect,
            )
            value = F.grid_sample(
                self.depth_gpu, grid_final, mode="bilinear",
                padding_mode="border", align_corners=False,
            )
            return result_gluv_x, result_gluv_y, value

        # ── PyTorch fallback (zero CPU sync) ───────────────────────────
        int_z_val = 1.0

        # Pre-compute direction vectors (const across iterations)
        dir_x = int_x - orig_x                             # (H, W)
        dir_y = int_y - orig_y                              # (H, W)

        # Pre-allocate per-pixel state
        has_hit = torch.zeros(render_h, render_w, dtype=torch.bool, device=dev)
        walk_at_hit = torch.zeros(render_h, render_w, device=dev)

        # Pre-allocate reusable tensors
        grid = torch.empty(1, render_h, render_w, 2, device=dev)
        pt_x = torch.empty(render_h, render_w, device=dev)
        pt_y = torch.empty(render_h, render_w, device=dev)

        # --- Forward pass (matches GLSL Stage 0 exactly) ---------------------
        # GLSL: walk starts at 0; check walk>1.0 BEFORE incrementing
        walk_f = 0.0
        for _ in range(1000):
            if walk_f > 1.0:
                break
            walk_f += probe_step

            mix_t_f = safe + (1.0 - safe) * walk_f
            ceiling_f = 1.0 - (orig_z_val * (1.0 - mix_t_f) + int_z_val * mix_t_f)

            torch.add(orig_x, dir_x, alpha=mix_t_f, out=pt_x)
            torch.add(orig_y, dir_y, alpha=mix_t_f, out=pt_y)

            if mirror:
                gx = want_aspect * _triangle_wave_t(pt_x, 4.0 * want_aspect)
                gy = _triangle_wave_t(pt_y, 4.0)
                grid[0, :, :, 0] = gx * scale_x
                grid[0, :, :, 1] = -gy
            else:
                grid[0, :, :, 0] = pt_x * scale_x
                grid[0, :, :, 1] = -pt_y

            sampled = F.grid_sample(
                self.depth_gpu, grid, mode="bilinear",
                padding_mode="border", align_corners=False,
            )
            d_val = sampled.view(render_h, render_w)
            surface = df_height * (
                d_val * (1.0 - df_invert) + (1.0 - d_val) * df_invert
            )

            newly_hit = (~has_hit) & (surface > ceiling_f)
            walk_at_hit = torch.where(newly_hit, walk_f, walk_at_hit)
            has_hit = has_hit | newly_hit

        # --- Backward linear refinement (matches GLSL Stage 1 exactly) ---
        quality_step = 1.0 / (200.0 + 1800.0 * quality_norm)

        # For no-hit pixels, walk stays at 1.0 (→ intersect)
        walk_result = torch.where(has_hit, walk_at_hit, torch.ones_like(walk_at_hit))
        refining = has_hit.clone()

        # Match GLSL: up to 1000 backward iterations (early exit when all done)
        for _ in range(1000):
            walk_cand = walk_result - quality_step
            # GLSL doesn't clamp < 0, but safe minimum
            walk_cand = walk_cand.clamp(min=-0.1)
            mix_t = safe + (1.0 - safe) * walk_cand
            ceiling = 1.0 - (orig_z_val * (1.0 - mix_t) + int_z_val * mix_t)

            pt_x = orig_x + dir_x * mix_t
            pt_y = orig_y + dir_y * mix_t

            if mirror:
                gx = want_aspect * _triangle_wave_t(pt_x, 4.0 * want_aspect)
                gy = _triangle_wave_t(pt_y, 4.0)
                grid[0, :, :, 0] = gx * scale_x
                grid[0, :, :, 1] = -gy
            else:
                grid[0, :, :, 0] = pt_x * scale_x
                grid[0, :, :, 1] = -pt_y

            sampled = F.grid_sample(
                self.depth_gpu, grid, mode="bilinear",
                padding_mode="border", align_corners=False,
            )
            d_val = sampled.view(render_h, render_w)
            surface = df_height * (
                d_val * (1.0 - df_invert) + (1.0 - d_val) * df_invert
            )
            # GLSL: ceiling >= surface → outside → stop
            still_inside = refining & (surface > ceiling)
            walk_result = torch.where(still_inside, walk_cand, walk_result)
            refining = still_inside

            if not refining.any():
                break

        walk_lo = walk_result.clamp(min=0.0)

        # Final result — matches GLSL: no-hit pixels have walk_lo=1.0 → intersect
        mix_t = safe + (1.0 - safe) * walk_lo
        result_gluv_x = orig_x + dir_x * mix_t
        result_gluv_y = orig_y + dir_y * mix_t

        # Depth at result for blur post-processing → (1, 1, H, W)
        if mirror:
            gx = want_aspect * _triangle_wave_t(result_gluv_x, 4.0 * want_aspect)
            gy = _triangle_wave_t(result_gluv_y, 4.0)
        else:
            gx, gy = result_gluv_x, result_gluv_y
        grid[0, :, :, 0] = gx * scale_x
        grid[0, :, :, 1] = -gy
        value = F.grid_sample(
            self.depth_gpu, grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        return result_gluv_x, result_gluv_y, value

    # ------------------------------------------------------------------ render

    @torch.inference_mode()
    def render_frame(
        self,
        render_w: int,
        render_h: int,
        state: DepthFlowState,
        quality_pct: float = 50.0,
        enable_inpaint: bool = True,
        inpaint_threshold: float = 0.04,
        inpaint_iterations: int = 6,
        inpaint_depth_aware: bool = True,
        enable_aa: bool = True,
    ) -> torch.Tensor:
        """Render one frame.  Returns (H, W, 3) uint8 tensor on CPU."""
        dev = self.device
        quality_norm = quality_pct / 100.0

        # --- pixel grids -------------------------------------------------
        gluv_x, gluv_y = self._make_gluv_grid(render_w, render_h, dev)
        aspect = float(render_w) / float(render_h)
        want_aspect = aspect

        # Scale gluv_x to match GLSL vertex shader: gluv.x ∈ [-aspect, +aspect]
        gluv_x = gluv_x * want_aspect

        mirror = state.mirror

        # DepthFlow state params
        df_height = state.height
        df_steady = state.steady
        df_focus = state.focus
        df_zoom = state.zoom
        df_iso = state.isometric
        df_dolly = state.dolly
        df_invert = state.invert
        offset_x = state.offset_x
        offset_y = state.offset_y
        center_x = state.center_x
        center_y = state.center_y
        origin_x = state.origin_x
        origin_y = state.origin_y

        # --- Camera projection (perspective, mode=2D) --------------------
        # Camera defaults: position=(0,0,0), fwd=(0,0,1), right=(1,0,0),
        #   up=(0,1,0), zoom=1, iso=0, dolly=0, focal=1, orbital=0
        # DepthMake injects: position.xy+=offset, iso+=..., dolly+=...,
        #   zoom+=(df_zoom-1), focal_length=(1 - focus*height)
        rel_focus = df_focus * df_height
        rel_steady = df_steady * df_height

        cam_pos_x = offset_x
        cam_pos_y = offset_y
        cam_zoom = df_zoom  # camera.zoom(1) + (df_zoom - 1) = df_zoom
        cam_iso = df_iso
        cam_dolly = df_dolly
        cam_focal = 1.0 - rel_focus

        # CameraRayOrigin (perspective):
        #   = pos + rect(gluv, zoom*iso) + backward*(orbital+dolly)
        #   backward = (0,0,-1)
        origin_cam_x = cam_pos_x + gluv_x * cam_zoom * cam_iso
        origin_cam_y = cam_pos_y + gluv_y * cam_zoom * cam_iso
        origin_cam_z = torch.full_like(gluv_x, -(cam_dolly))  # orbital=0

        # CameraRayTarget (perspective):
        #   = pos + rect(gluv, zoom) + backward*orbital + forward*focal
        target_cam_x = cam_pos_x + gluv_x * cam_zoom
        target_cam_y = cam_pos_y + gluv_y * cam_zoom
        target_cam_z = torch.full_like(gluv_x, cam_focal)  # -orbital + focal

        # CameraRay2D — line-plane intersection with plane z=1
        #   plane_point=(0,0,1), plane_normal=(0,0,1)
        num = 1.0 - origin_cam_z  # dot((0,0,1) - origin, (0,0,1))
        den = target_cam_z - origin_cam_z  # dot(target-origin, (0,0,1))
        # Avoid division by zero
        den = torch.where(den.abs() < 1e-10, torch.full_like(den, 1e-10), den)
        t_plane = num / den

        # intersection = origin + t * (target - origin)
        cam_gluv_x = origin_cam_x + t_plane * (target_cam_x - origin_cam_x)
        cam_gluv_y = origin_cam_y + t_plane * (target_cam_y - origin_cam_y)

        # OOB check
        oob = (t_plane < 0) | (gluv_x.abs() > want_aspect)

        # --- DepthMake: shift origin & build intersect -------------------
        # camera.origin += (depth.origin, 0)
        orig_x = origin_cam_x + origin_x
        orig_y = origin_cam_y + origin_y
        orig_z = origin_cam_z  # z unchanged

        # intersect = (center + cam.gluv, 1) - (position, 0) / (1-rel_steady)
        #   when glued=True
        denom_steady = max(1.0 - rel_steady, 1e-8)
        intersect_x = center_x + cam_gluv_x - cam_pos_x / denom_steady
        intersect_y = center_y + cam_gluv_y - cam_pos_y / denom_steady
        intersect_z = torch.ones_like(gluv_x)

        # --- Ray marching: Batched Forward + Binary Search ----------------
        result_gluv_x, result_gluv_y, value = self._ray_march_v2(
            orig_x, orig_y, orig_z,
            intersect_x, intersect_y, intersect_z,
            cam_gluv_x, cam_gluv_y,
            df_height, df_invert, mirror, want_aspect,
            quality_norm, render_h, render_w,
        )

        # --- Sample image at result gluv ---------------------------------
        img_grid = self._gluv_to_grid(result_gluv_x, result_gluv_y,
                                       mirror, want_aspect)
        color = F.grid_sample(self.image_gpu, img_grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
        # color: (1, 3, H, W)

        # Apply OOB mask
        oob_mask = oob.unsqueeze(0).unsqueeze(0)
        color = torch.where(oob_mask, torch.zeros_like(color), color)

        # --- Disocclusion inpaint (before post-processing / AA) -----------
        if enable_inpaint and self._should_inpaint(state):
            color = self._inpaint_disocclusions(
                color=color,
                result_gluv_x=result_gluv_x,
                result_gluv_y=result_gluv_y,
                depth_value=value,
                threshold=inpaint_threshold,
                iterations=inpaint_iterations,
                depth_aware=inpaint_depth_aware,
            )


        # --- Post-processing (matching GLSL) ------------------------------
        r, g, b = color[:, 0:1], color[:, 1:2], color[:, 2:3]

        # Lens distortion
        if state.lens_enable:
            agluv_x = gluv_x / aspect
            agluv_y = gluv_y
            length_agluv = (agluv_x ** 2 + agluv_y ** 2).sqrt()
            safe_len = length_agluv.clamp(min=1e-8)
            norm_x = agluv_x / safe_len
            norm_y = agluv_y / safe_len

            decay = (0.62 * length_agluv).pow(10.0 - 9.0 * state.lens_decay)
            delta_x = 0.5 * state.lens_intensity * norm_x * decay
            delta_y = 0.5 * state.lens_intensity * norm_y * decay

            acc_r = torch.zeros_like(r)
            acc_g = torch.zeros_like(g)
            acc_b = torch.zeros_like(b)
            n_samples = state.lens_quality
            for i in range(n_samples):
                frac = float(i) / float(n_samples)
                gx_r = result_gluv_x - 1.0 * frac * delta_x
                gy_r = result_gluv_y - 1.0 * frac * delta_y
                gx_g = result_gluv_x - 2.0 * frac * delta_x
                gy_g = result_gluv_y - 2.0 * frac * delta_y
                gx_b = result_gluv_x - 4.0 * frac * delta_x
                gy_b = result_gluv_y - 4.0 * frac * delta_y

                gr = self._gluv_to_grid(gx_r, gy_r, mirror, want_aspect)
                gg = self._gluv_to_grid(gx_g, gy_g, mirror, want_aspect)
                gb = self._gluv_to_grid(gx_b, gy_b, mirror, want_aspect)
                sr = F.grid_sample(self.image_gpu, gr, mode="bilinear",
                                   padding_mode="border", align_corners=False)
                sg = F.grid_sample(self.image_gpu, gg, mode="bilinear",
                                   padding_mode="border", align_corners=False)
                sb = F.grid_sample(self.image_gpu, gb, mode="bilinear",
                                   padding_mode="border", align_corners=False)
                acc_r += sr[:, 0:1]
                acc_g += sg[:, 1:2]
                acc_b += sb[:, 2:3]

            r = acc_r / n_samples
            g = acc_g / n_samples
            b = acc_b / n_samples

        # Depth-of-field blur
        elif state.blur_enable:
            depth_val = value.squeeze(0)  # (1, H, W)
            smoothstep_val = _smoothstep_t(
                state.blur_start, state.blur_end, 1.0 - depth_val,
            )
            intensity_map = state.blur_intensity * smoothstep_val.pow(state.blur_exponent)
            acc_color = color.clone()
            n_blur_samples = state.blur_directions * state.blur_quality
            tau_val = 2.0 * math.pi
            for d in range(state.blur_directions):
                angle = tau_val * d / state.blur_directions
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                for q in range(1, state.blur_quality + 1):
                    w = float(q) / float(state.blur_quality)
                    dx = cos_a * w * intensity_map
                    dy = sin_a * w * intensity_map
                    gx = result_gluv_x + dx.squeeze(0)
                    gy = result_gluv_y + dy.squeeze(0)
                    gr = self._gluv_to_grid(gx, gy, mirror, want_aspect)
                    s = F.grid_sample(self.image_gpu, gr, mode="bilinear",
                                      padding_mode="border", align_corners=False)
                    acc_color += s
            fused = acc_color / n_blur_samples
            r, g, b = fused[:, 0:1], fused[:, 1:2], fused[:, 2:3]

        # Vignette
        if state.vig_enable:
            # astuv coordinates [0,1] — aspect-corrected (matching GLSL)
            astuv_x = (gluv_x / want_aspect + 1.0) / 2.0
            astuv_y = (gluv_y + 1.0) / 2.0
            away_x = astuv_x * (1.0 - astuv_x)
            away_y = astuv_y * (1.0 - astuv_y)
            linear_v = state.vig_decay * away_x * away_y
            vig_mult = linear_v.pow(state.vig_intensity).clamp(0, 1)
            vig_mult = vig_mult.unsqueeze(0).unsqueeze(0)
            r = r * vig_mult
            g = g * vig_mult
            b = b * vig_mult

        # Color adjustments
        if state.color_saturation != 1.0:
            h, s, v = _rgb_to_hsv(r, g, b)
            s = (s * state.color_saturation).clamp(0, 1)
            r, g, b = _hsv_to_rgb(h, s, v)

        if state.color_contrast != 1.0:
            r = ((r - 0.5) * state.color_contrast + 0.5).clamp(0, 1)
            g = ((g - 0.5) * state.color_contrast + 0.5).clamp(0, 1)
            b = ((b - 0.5) * state.color_contrast + 0.5).clamp(0, 1)

        if state.color_brightness != 1.0:
            r = (r * state.color_brightness).clamp(0, 1)
            g = (g * state.color_brightness).clamp(0, 1)
            b = (b * state.color_brightness).clamp(0, 1)

        if state.color_gamma != 1.0:
            inv_g = 1.0 / state.color_gamma
            r = r.clamp(min=0).pow(inv_g)
            g = g.clamp(min=0).pow(inv_g)
            b = b.clamp(min=0).pow(inv_g)

        if state.color_sepia > 0:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            r = r * (1 - state.color_sepia) + lum * 1.2 * state.color_sepia
            g = g * (1 - state.color_sepia) + lum * 1.0 * state.color_sepia
            b = b * (1 - state.color_sepia) + lum * 0.8 * state.color_sepia

        if state.color_grayscale > 0:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            r = r * (1 - state.color_grayscale) + lum * state.color_grayscale
            g = g * (1 - state.color_grayscale) + lum * state.color_grayscale
            b = b * (1 - state.color_grayscale) + lum * state.color_grayscale

        # Reassemble (1, 3, H, W) → (H, W, 3) uint8
        frame = torch.cat([r, g, b], dim=1).squeeze(0)  # (3, H, W)

        # Match ShaderFlow's default subsample=2 final pass:
        # 2×2 subpixel averaging with bilinear — equivalent to a
        # separable [1, 6, 1]/8 tent filter (centre ≈ 56%).
        if enable_aa:
            _aa_k = torch.tensor([1.0, 6.0, 1.0], device=frame.device) / 8.0
            _aa_kh = _aa_k.view(1, 1, 1, 3).expand(3, 1, 1, 3)   # horizontal
            _aa_kv = _aa_k.view(1, 1, 3, 1).expand(3, 1, 3, 1)   # vertical
            f4d = frame.unsqueeze(0)                               # (1, 3, H, W)
            f4d = F.conv2d(F.pad(f4d, (1, 1, 0, 0), mode="replicate"),
                           _aa_kh, groups=3)
            f4d = F.conv2d(F.pad(f4d, (0, 0, 1, 1), mode="replicate"),
                           _aa_kv, groups=3)
            frame = f4d.squeeze(0)                                 # (3, H, W)

        frame = frame.clamp(0, 1).mul(255).byte()
        frame = frame.permute(1, 2, 0).contiguous()      # (H, W, 3)
        return frame.cpu()

    # -------------------------------------------------------- inpainting

    @staticmethod
    def _should_inpaint(state: "DepthFlowState") -> bool:
        """Return True if the current state has any camera movement that
        would produce disocclusion artifacts worth inpainting."""
        return (
            abs(state.offset_x) > 1e-5 or
            abs(state.offset_y) > 1e-5 or
            abs(state.height - 0.2) > 1e-5 or
            abs(state.isometric) > 1e-5 or
            abs(state.dolly) > 1e-5 or
            abs(state.zoom - 1.0) > 1e-5
        )

    @staticmethod
    @torch.inference_mode()
    def _inpaint_disocclusions(
        color: "torch.Tensor",             # (1, 3, H, W) float [0,1]
        result_gluv_x: "torch.Tensor",     # (H, W)
        result_gluv_y: "torch.Tensor",     # (H, W)
        depth_value: Optional["torch.Tensor"] = None,  # (1, 1, H, W) float
        threshold: float = 0.04,
        iterations: int = 6,
        depth_aware: bool = True,
    ) -> "torch.Tensor":
        """
        Fill disocclusion 'holes' at foreground silhouette edges.

        When the camera moves, previously-occluded background regions become
        visible near the foreground edge.  The ray-march has no colour source
        for those pixels and replicates neighbours, producing stretched /
        smeared edges.

        Algorithm:
        1. UV gradient mask — detect pixels where the UV coordinate gradient
           is abnormally large (disocclusion boundaries).
        2. Depth-aware mask — combine with depth edge detection to reduce
           false positives on textured flat surfaces.
        3. Mask dilation — expand the mask slightly to catch fringe pixels.
        4. Iterative fill — weighted-average from valid (non-disoccluded)
           neighbours via 3×3 conv2d, with confidence blending.
        5. Foreground protection — holes don't contribute to the average;
           only valid neighbours participate.
        """
        dev = color.device

        # ── 1. UV gradient mask ──────────────────────────────────────────
        # Forward finite differences
        dx = F.pad(
            (result_gluv_x[:, 1:] - result_gluv_x[:, :-1]).unsqueeze(0).unsqueeze(0),
            (0, 1, 0, 0),
        ).squeeze()
        dy = F.pad(
            (result_gluv_y[1:, :] - result_gluv_y[:-1, :]).unsqueeze(0).unsqueeze(0),
            (0, 0, 0, 1),
        ).squeeze()
        uv_grad = dx.abs() + dy.abs()  # (H, W)
        steep = uv_grad > threshold    # (H, W) bool

        # ── 2. Depth-aware mask ──────────────────────────────────────────
        if depth_aware and depth_value is not None:
            dv = depth_value.squeeze()  # (H, W)
            # Depth gradient
            d_dx = F.pad(
                (dv[:, 1:] - dv[:, :-1]).unsqueeze(0).unsqueeze(0),
                (0, 1, 0, 0),
            ).squeeze()
            d_dy = F.pad(
                (dv[1:, :] - dv[:-1, :]).unsqueeze(0).unsqueeze(0),
                (0, 0, 0, 1),
            ).squeeze()
            depth_grad = d_dx.abs() + d_dy.abs()  # (H, W)
            # Depth edge: where gradient is significantly above average
            depth_mean = depth_grad.mean()
            depth_std = depth_grad.std()
            # Aggressive detection: Union of UV tears and Depth edges
            # Also catch pixels where depth is significantly stretched
            steep = steep | depth_edge


        if not steep.any():
            return color

        # ── 3. Mask dilation ─────────────────────────────────────────────
        # Dilate aggressively based on iteration count to swallow long streaks
        dial_radius = max(2, iterations // 3)
        steep_float = steep.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        steep = F.max_pool2d(steep_float, kernel_size=dial_radius*2+1, stride=1, padding=dial_radius)
        steep = steep.squeeze() > 0.5  # back to (H, W) bool


        # ── 4. Iterative fill ────────────────────────────────────────────
        # 3×3 uniform kernel (shared across channels via groups)
        fill_k = torch.ones(1, 1, 3, 3, dtype=color.dtype, device=dev) / 9.0
        valid = (~steep).float()  # (H, W): 1=valid, 0=hole
        c = color.clone()         # (1, 3, H, W)

        for _ in range(iterations):
            if not steep.any():
                break

            valid_4d = valid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            # Background-prioritized fill:
            # We want to sample background colors for sky holes, not hair colors.
            # Use depth_value to weight neighbors: deeper pixels (lower values) get more weight.
            if depth_aware and depth_value is not None:
                dv_4d = depth_value.detach()
                # Weight = e^(-5*depth) * valid_mask. In DepthFlow, 0=bg, 1=fg.
                # So we want low depth values to have high weight.
                d_weight = torch.exp(-5.0 * dv_4d) * valid_4d
                nb_weight = F.conv2d(F.pad(d_weight, (1, 1, 1, 1), mode="constant", value=0), fill_k)
                nb_color = F.conv2d(
                    F.pad(c * d_weight, (1, 1, 1, 1), mode="constant", value=0),
                    fill_k.expand(3, 1, 3, 3), groups=3
                )
            else:
                nb_weight = F.conv2d(F.pad(valid_4d, (1, 1, 1, 1), mode="constant", value=0), fill_k)
                nb_color = F.conv2d(
                    F.pad(c * valid_4d, (1, 1, 1, 1), mode="constant", value=0),
                    fill_k.expand(3, 1, 3, 3), groups=3
                )

            # Normalize by weights
            safe_weight = nb_weight.clamp(min=1e-8)
            nb_color = nb_color / safe_weight

            # ── 5. Confidence blend ──────────────────────────────────────
            # Only fill steep pixels that have at least one valid neighbour
            has_valid_nb = nb_weight.squeeze() > (1.0 / 9.0 - 1e-6)  # at least ~1 valid
            fillable = steep & has_valid_nb  # (H,W)

            # Confidence: how many valid neighbours we had (0→1)
            # More valid neighbours → higher confidence in the fill
            confidence = nb_weight.squeeze().clamp(0.0, 1.0)  # (H,W)
            confidence_4d = confidence.unsqueeze(0).unsqueeze(0).expand_as(c)
            fill_4d = fillable.unsqueeze(0).unsqueeze(0).expand_as(c)

            # Blend: filled_color * confidence + original * (1 - confidence)
            blended = nb_color * confidence_4d + c * (1.0 - confidence_4d)
            c = torch.where(fill_4d, blended, c)

            # ── 6. Update valid mask ─────────────────────────────────────
            # Newly-filled pixels become valid for next iteration
            valid = (valid + fillable.float()).clamp(0.0, 1.0)
            steep = steep & ~fillable

        return c

    # ------------------------------------------------------------------ video

    def render_video(
        self,
        output_path: str,
        render_w: int,
        render_h: int,
        fps: float = 60.0,
        duration: float = 5.0,
        ssaa: float = 1.0,
        quality_pct: float = 50.0,
        camera_movement: str = "horizontal",
        intensity: float = 1.0,
        smooth: bool = True,
        loop: bool = True,
        reverse: bool = False,
        phase: float = 0.0,
        steady_depth: float = 0.3,
        isometric_val: float = 0.6,
        codec: str = "h264_nvenc",
        output_format: str = "mp4",
        progress_cb: Optional[Callable[[int, int], None]] = None,
        capture_frames: int = 0,
        enable_inpaint: bool = True,
        inpaint_threshold: float = 0.04,
        inpaint_iterations: int = 6,
        inpaint_depth_aware: bool = True,
        enable_aa: bool = True,
    ) -> str:
        """Render a full parallax video to *output_path*.

        If *capture_frames* > 0, also collect up to that many frames as
        tensors and store them in ``self.captured_frames`` (list of HWC uint8
        CPU tensors).  Set to -1 to capture all frames.
        """

        # SSAA: render at higher resolution, then downscale
        ssaa_w = int(render_w * ssaa)
        ssaa_h = int(render_h * ssaa)
        total_frames = max(1, int(duration * fps))

        # Build FFmpeg command
        ffmpeg_bin = _find_ffmpeg()
        vcodec = _pick_codec(codec, ffmpeg_bin)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{ssaa_w}x{ssaa_h}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]
        # Downscale from SSAA resolution to target
        if ssaa != 1.0:
            cmd += ["-vf", f"scale={render_w}:{render_h}:flags=lanczos"]
        cmd += [
            "-c:v", vcodec,
            "-pix_fmt", "yuv420p",
            "-an",
        ]
        # Codec-specific quality settings
        if "nvenc" in vcodec:
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "18"]
        elif vcodec == "libx264":
            cmd += ["-preset", "fast", "-crf", "18"]
        elif vcodec == "libx265":
            cmd += ["-preset", "fast", "-crf", "20"]

        cmd.append(output_path)

        print(f"[DepthFlow CUDA] Rendering {total_frames} frames "
              f"@ {ssaa_w}×{ssaa_h} (SSAA {ssaa}x) → {render_w}×{render_h}")
        print(f"[DepthFlow CUDA] Codec: {vcodec}, Output: {output_path}")
        print(f"[DepthFlow CUDA] inpaint={enable_inpaint}, aa={enable_aa}, "
              f"threshold={inpaint_threshold}, iterations={inpaint_iterations}, "
              f"depth_aware={inpaint_depth_aware}")

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.perf_counter()

        # Optional frame capture (avoids re-decoding the video later)
        do_capture = capture_frames != 0
        max_cap = total_frames if capture_frames < 0 else capture_frames
        self.captured_frames: list = []

        try:
            for frame_idx in range(total_frames):
                tau = frame_idx / max(total_frames - 1, 1)
                state = compute_animation_state(
                    camera_movement, tau,
                    intensity=intensity, smooth=smooth, loop=loop,
                    reverse=reverse, phase=phase,
                    steady_depth=steady_depth, isometric=isometric_val,
                )
                frame = self.render_frame(
                    ssaa_w, ssaa_h, state, quality_pct,
                    enable_inpaint=enable_inpaint,
                    inpaint_threshold=inpaint_threshold,
                    inpaint_iterations=inpaint_iterations,
                    inpaint_depth_aware=inpaint_depth_aware,
                    enable_aa=enable_aa,
                )
                proc.stdin.write(frame.numpy().tobytes())

                # Capture frame directly (already CPU uint8 HWC)
                if do_capture and len(self.captured_frames) < max_cap:
                    self.captured_frames.append(frame)

                if progress_cb:
                    progress_cb(frame_idx + 1, total_frames)
                if (frame_idx + 1) % 60 == 0 or frame_idx == total_frames - 1:
                    elapsed = time.perf_counter() - t0
                    render_fps = (frame_idx + 1) / max(elapsed, 0.001)
                    print(f"[DepthFlow CUDA] {frame_idx+1}/{total_frames} "
                          f"({render_fps:.1f} fps)")
        finally:
            proc.stdin.close()
            proc.wait()

        elapsed = time.perf_counter() - t0
        print(f"[DepthFlow CUDA] Done in {elapsed:.1f}s "
              f"({total_frames/max(elapsed,0.001):.1f} avg fps)")
        return output_path


# ────────────────────────────────────────────────────────────────────────────
# Tensor-level triangle wave (vectorised)
# ────────────────────────────────────────────────────────────────────────────

def _triangle_wave_t(x: torch.Tensor, period: float) -> torch.Tensor:
    """``triangle_wave`` for whole tensors."""
    return 2.0 * (((2.0 * x / period - 0.5) % 2.0) - 1.0).abs() - 1.0


# ────────────────────────────────────────────────────────────────────────────
# FFmpeg utilities
# ────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    for p in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).is_file():
            return p
    return "ffmpeg"


def _pick_codec(requested: str, ffmpeg_bin: str) -> str:
    """Validate that the requested codec is available, else fall back."""
    codec_map = {
        "h264_nvenc": "h264_nvenc", "h264-nvenc": "h264_nvenc",
        "h265_nvenc": "hevc_nvenc", "h265-nvenc": "hevc_nvenc",
        "hevc_nvenc": "hevc_nvenc",
        "h264": "libx264", "libx264": "libx264",
        "h265": "libx265", "libx265": "libx265",
        "av1-svt": "libsvtav1", "libsvtav1": "libsvtav1",
    }
    vcodec = codec_map.get(requested.lower().strip(), "libx264")

    # Quick check — try encoding 1 frame
    if "nvenc" in vcodec:
        try:
            r = subprocess.run(
                [ffmpeg_bin, "-y", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x16", "-i", "pipe:0",
                 "-frames:v", "1", "-c:v", vcodec, "-f", "null", "-"],
                input=b"\x00" * (16 * 16 * 3),
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                print(f"[DepthFlow CUDA] {vcodec} unavailable, falling back to libx264")
                vcodec = "libx264"
        except Exception:
            vcodec = "libx264"
    return vcodec


# ────────────────────────────────────────────────────────────────────────────
# Public convenience
# ────────────────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if CUDA rendering is feasible."""
    return _check_cuda()
