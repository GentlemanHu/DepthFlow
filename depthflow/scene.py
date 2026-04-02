import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Optional

import imageio.v3 as imageio
import pooch
from attrs import Factory, define
from cyclopts import Parameter
from imgui_bundle import imgui
from shaderflow.message import ShaderMessage
from shaderflow.scene import ShaderScene
from shaderflow.texture import ShaderTexture
from shaderflow.variable import ShaderVariable

import depthflow
from depthflow.animation import (
    Animation,
    DepthAnimation,
    FilterBase,
    PresetBase,
)
from depthflow.estimators import DepthEstimator
from depthflow.estimators.anything import (
    DepthAnythingV1,
    DepthAnythingV2,
    DepthAnythingV3,
)
from depthflow.estimators.depthpro import DepthPro
from depthflow.estimators.marigold import Marigold
from depthflow.estimators.zoedepth import ZoeDepth
from depthflow.state import DepthState


class Assets:
    """Copyright property of the original owners"""

    def background() -> Path:
        return pooch.retrieve(
            url="https://w.wallhaven.cc/full/pk/wallhaven-pkz5r9.png",
            known_hash="xxh128:6fe8d585cfc4b8fc623b5450d06bcdc4",
            path=depthflow.directories.user_data_path,
            fname="wallhaven-pkz5r9.png",
            progressbar=True,
        )

# ---------------------------------------------------------------------------- #

@define
class DepthScene(ShaderScene):
    state:     DepthState     = Factory(DepthState)
    estimator: DepthEstimator = Factory(DepthAnythingV2)
    animation: DepthAnimation = Factory(DepthAnimation)

    def smartset(self, object: Any) -> Any:
        if isinstance(object, DepthEstimator):
            self.estimator = object
        return object

    # ------------------------------------------------------------------------ #
    # Command line interface

    def commands(self):
        self.cli.help = depthflow.__about__
        self.cli.version = depthflow.__version__
        self.cli.command(self.input)

        with contextlib.nullcontext("🌊 Depth Estimator") as group:
            options = dict(group=group, result_action=self.smartset)
            self.cli.command(DepthAnythingV1, name="da1",      **options)
            self.cli.command(DepthAnythingV2, name="da2",      **options)
            self.cli.command(DepthAnythingV3, name="da3",      **options)
            self.cli.command(DepthPro,        name="depthpro", **options)
            self.cli.command(ZoeDepth,        name="zoedepth", **options)
            self.cli.command(Marigold,        name="marigold", **options)

        with contextlib.nullcontext("🎬 Animation presets") as group:
            for preset in Animation.members():
                if issubclass(preset, PresetBase):
                    self.cli.command(preset, group=group, result_action=self.animation.add)

        with contextlib.nullcontext("🎨 Post-processing") as group:
            for post in Animation.members():
                if issubclass(post, FilterBase):
                    self.cli.command(post, group=group, result_action=self.animation.add)

    def input(self,
        image: Annotated[Optional[str], Parameter(
            help="Input image from Path, NumPy, URL (None to default)",
            name=("--image", "-i"))] = None,
        depth: Annotated[Optional[str], Parameter(
            help="Input depthmap of the image (None to estimate)",
            name=("--depth", "-d"))] = None,
    ) -> None:
        """Use the given image(s) and depthmap(s) as the input of the scene"""
        if (image is None):
            image = Assets.background()

        # Load estimate input image
        image = imageio.imread(image)
        depth = imageio.imread(depth) \
            if (depth is not None) else \
            self.estimator.estimate(image)

        self.image.from_numpy(image)
        self.depth.from_numpy(depth)

        # Keep raw numpy copies for CUDA backend
        self._raw_image = image
        self._raw_depth = depth

        # Match rendering resolution to image
        self.resolution = self.image.size

    # ------------------------------------------------------------------------ #
    # CUDA backend — bypass OpenGL entirely

    def cuda_render(
        self,
        output: Path | str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        time: float | None = None,
        quality: float = 50.0,
        ssaa: float = 1.0,
        codec: str = "h264_nvenc",
        format: str = "mp4",
    ) -> Path:
        """Render using CUDA/PyTorch instead of OpenGL/ShaderFlow."""
        from depthflow.cuda_renderer import (
            CudaDepthFlowRenderer,
            DepthFlowState,
            compute_animation_state,
            is_available,
        )

        if not is_available():
            raise RuntimeError("CUDA not available for DepthFlow rendering")

        # Ensure inputs are loaded
        if not hasattr(self, "_raw_image"):
            self.input()

        import numpy as np
        img = np.asarray(self._raw_image, dtype=np.float32)
        dep = np.asarray(self._raw_depth, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        if dep.max() > 1.5:
            dep = dep / 255.0
        if dep.ndim == 3:
            dep = dep[..., 0]

        renderer = CudaDepthFlowRenderer(img, dep)
        duration = time or self.runtime or 5.0
        output = Path(output).with_suffix(f".{format}")

        # Determine animation type and params from self.animation.steps
        move_type, move_params = self._extract_animation_params()

        renderer.render_video(
            output_path=str(output),
            render_w=width,
            render_h=height,
            fps=fps,
            duration=duration,
            ssaa=ssaa,
            quality_pct=quality,
            codec=codec,
            output_format=format,
            **move_params,
        )
        return output

    def _extract_animation_params(self) -> tuple[str, dict]:
        """Extract camera_movement type and params from animation steps."""
        defaults = dict(
            camera_movement="orbital",
            intensity=1.0, smooth=True, loop=True,
            reverse=False, phase=0.0,
            steady_depth=0.3, isometric_val=0.6,
        )
        for step in self.animation.steps:
            cls_name = type(step).__name__
            name_map = {
                "Vertical": "vertical", "Horizontal": "horizontal",
                "Zoom": "zoom", "Circle": "circle",
                "Dolly": "dolly", "Orbital": "orbital",
            }
            if cls_name in name_map:
                defaults["camera_movement"] = name_map[cls_name]
                defaults["intensity"] = getattr(step, "intensity", 1.0)
                defaults["reverse"] = getattr(step, "reverse", False)
                defaults["smooth"] = getattr(step, "smooth", True)
                defaults["loop"] = getattr(step, "loop", True)
                defaults["phase"] = getattr(step, "phase", 0.0)
                defaults["steady_depth"] = getattr(step, "steady", 0.3)
                defaults["isometric_val"] = getattr(step, "isometric", 0.6)
                break
        return defaults["camera_movement"], defaults

    # ------------------------------------------------------------------------ #
    # Module implementation

    def build(self) -> None:
        self.depth = ShaderTexture(scene=self, name="depth", anisotropy=1).repeat(False)
        self.image = ShaderTexture(scene=self, name="image").repeat(False)
        self.shader.fragment = (depthflow.resources/"depthflow.glsl")
        self.runtime = 5.0

    def setup(self) -> None:
        if (not self.animation.steps):
            self.animation.add(Animation.Orbital())
        if self.image.is_empty():
            self.input()

    def update(self) -> None:
        self.animation.apply(self)

    def handle(self, message: ShaderMessage) -> None:
        ShaderScene.handle(self, message)

        if isinstance(message, ShaderMessage.Window.FileDrop):
            self.input(image=message.first, depth=message.second)
            self._load_inputs()

    def pipeline(self) -> Iterable[ShaderVariable]:
        yield from ShaderScene.pipeline(self)
        yield from self.state.pipeline()

    # ------------------------------------------------------------------------ #

    def ui(self) -> None:
        if (state := imgui.slider_float("Height", self.state.height, 0, 1, "%.2f"))[0]:
            self.state.height = state[1]
        if (state := imgui.slider_float("Steady", self.state.steady, 0, 1, "%.2f"))[0]:
            self.state.steady = state[1]
        if (state := imgui.slider_float("Focus", self.state.focus, 0, 1, "%.2f"))[0]:
            self.state.focus = state[1]
        if (state := imgui.slider_float("Invert", self.state.invert, 0, 1, "%.2f"))[0]:
            self.state.invert = state[1]
        if (state := imgui.slider_float("Zoom", self.state.zoom, 0, 2, "%.2f"))[0]:
            self.state.zoom = state[1]
        if (state := imgui.slider_float("Isometric", self.state.isometric, 0, 1, "%.2f"))[0]:
            self.state.isometric = state[1]
        if (state := imgui.slider_float("Dolly", self.state.dolly, 0, 5, "%.2f"))[0]:
            self.state.dolly = state[1]

        imgui.text("- True camera position")
        if (state := imgui.slider_float("Center X", self.state.center_x, -self.aspect_ratio, self.aspect_ratio, "%.2f"))[0]:
            self.state.center_x = state[1]
        if (state := imgui.slider_float("Center Y", self.state.center_y, -1, 1, "%.2f"))[0]:
            self.state.center_y = state[1]

        imgui.text("- Fixed point at height changes")
        if (state := imgui.slider_float("Origin X", self.state.origin_x, -self.aspect_ratio, self.aspect_ratio, "%.2f"))[0]:
            self.state.origin_x = state[1]
        if (state := imgui.slider_float("Origin Y", self.state.origin_y, -1, 1, "%.2f"))[0]:
            self.state.origin_y = state[1]

        imgui.text("- Parallax offset")
        if (state := imgui.slider_float("Offset X", self.state.offset_x, -2, 2, "%.2f"))[0]:
            self.state.offset_x = state[1]
        if (state := imgui.slider_float("Offset Y", self.state.offset_y, -2, 2, "%.2f"))[0]:
            self.state.offset_y = state[1]
