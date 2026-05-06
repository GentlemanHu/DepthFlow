"""VHS-compatible DepthFlow animation presets.

This module keeps local ComfyUI/VHS command extensions separate from the
upstream animation module so upstream merges usually only touch scene.py.
"""
from __future__ import annotations

import math

from pydantic import Field

from depthflow.animation import Action
from depthflow.state import DepthState


def _cycle(time: float, reverse: bool) -> tuple[float, float]:
    tau = 1.0 - float(time) if reverse else float(time)
    return tau, math.tau * tau


def _sine(time: float, intensity: float, phase: float, reverse: bool = False) -> float:
    tau, _ = _cycle(time, reverse)
    return float(intensity) * math.sin(math.tau * (tau + float(phase)))


def _triangle(time: float, intensity: float, phase: float, reverse: bool = False) -> float:
    tau, _ = _cycle(time, reverse)
    return float(intensity) * (2.0 * abs(((tau + phase) % 1.0) - 0.5) - 0.5)


def _oscillate(time: float, intensity: float, phase: float, smooth: bool, reverse: bool) -> float:
    fn = _sine if smooth else _triangle
    return fn(time, intensity, phase, reverse)


class Horizontal(Action):
    """Apply a horizontal motion in offsets"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    smooth: bool = True
    loop: bool = True
    phase: float = Field(default=0.0, ge=-1.0, le=1.0)
    steady: float = Field(default=0.3, ge=-2.0, le=2.0)
    isometric: float = Field(default=0.6, ge=0.0, le=1.0)

    def apply(self, state: DepthState, time: float) -> None:
        state.steady = self.steady
        state.isometric = self.isometric
        phase = self.phase if self.loop else -0.25
        value = _oscillate(time, self.intensity, phase, self.smooth, self.reverse)
        state.offset = (value, state.offset[1])


class Vertical(Action):
    """Apply a vertical motion in offsets"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    smooth: bool = True
    loop: bool = True
    phase: float = Field(default=0.0, ge=-1.0, le=1.0)
    steady: float = Field(default=0.3, ge=-2.0, le=2.0)
    isometric: float = Field(default=0.6, ge=0.0, le=1.0)

    def apply(self, state: DepthState, time: float) -> None:
        state.steady = self.steady
        state.isometric = self.isometric
        phase = self.phase if self.loop else -0.25
        value = _oscillate(time, self.intensity, phase, self.smooth, self.reverse)
        state.offset = (state.offset[0], value)


class Circle(Action):
    """Apply a circular motion in offsets"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    phase: float = Field(default=0.0, ge=-1.0, le=1.0)
    steady: float = Field(default=0.3, ge=-2.0, le=2.0)
    isometric: float = Field(default=0.6, ge=0.0, le=1.0)

    def apply(self, state: DepthState, time: float) -> None:
        state.steady = self.steady
        state.isometric = self.isometric
        state.offset = (
            _sine(time, self.intensity * 0.5, self.phase, self.reverse),
            _sine(time, self.intensity * 0.5, self.phase + 0.25, self.reverse),
        )


class Zoom(Action):
    """Animate parallax height for a zoom-like depth push"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    smooth: bool = True
    loop: bool = True
    phase: float = Field(default=0.0, ge=-1.0, le=1.0)
    isometric: float = Field(default=0.6, ge=0.0, le=1.0)

    def apply(self, state: DepthState, time: float) -> None:
        state.isometric = self.isometric
        if self.loop:
            state.height = _oscillate(
                time, self.intensity * 0.5, self.phase, self.smooth, self.reverse,
            ) + self.intensity * 0.5
            return
        tau, _ = _cycle(time, self.reverse)
        state.height = max(0.0, min(2.0, tau * self.intensity))


class Dolly(Action):
    """Animate isometric depth for a dolly zoom effect"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    smooth: bool = True
    loop: bool = True
    phase: float = Field(default=0.0, ge=-1.0, le=1.0)
    steady: float = Field(default=0.3, ge=-2.0, le=2.0)

    def apply(self, state: DepthState, time: float) -> None:
        state.height = self.intensity / 3.0
        state.steady = self.steady
        state.focus = self.steady
        phase = self.phase + (0.75 if self.reverse else 0.25)
        if not self.loop:
            phase = self.phase + (-0.75 if self.reverse else 0.25)
        state.isometric = _oscillate(
            time, self.intensity * 0.5, phase, self.smooth, not self.reverse,
        ) + self.intensity * 0.5


class Orbital(Action):
    """Animate perspective and horizontal offset for an orbital move"""
    intensity: float = Field(default=1.0, ge=0.0, le=4.0)
    reverse: bool = False
    steady: float = Field(default=0.3, ge=-2.0, le=2.0)

    def apply(self, state: DepthState, time: float) -> None:
        _, cycle = _cycle(time, self.reverse)
        state.steady = self.steady
        state.focus = self.steady
        state.zoom = 0.98
        state.isometric = math.cos(cycle) * (self.intensity / 4.0) + self.intensity / 2.0 + 0.5
        state.offset = (math.sin(cycle) * (self.intensity / 4.0), state.offset[1])


VHS_ANIMATION_CLASSES = (Horizontal, Vertical, Circle, Zoom, Dolly, Orbital)
