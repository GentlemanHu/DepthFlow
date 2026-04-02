import sys
from typing import Annotated

import cyclopts
from cyclopts import App, Parameter


def scene(*ctx: Annotated[str, Parameter(
    allow_leading_hyphen=True,
    show=False,
)]) -> None:
    """🟢 Run depthflow's command line interface"""
    args = list(ctx)

    # Intercept --backend cuda before passing to ShaderFlow
    use_cuda = False
    if "--backend" in args:
        idx = args.index("--backend")
        if idx + 1 < len(args) and args[idx + 1].lower() == "cuda":
            use_cuda = True
            args.pop(idx)  # remove --backend
            args.pop(idx)  # remove cuda

    from depthflow.scene import DepthScene
    ds = DepthScene()

    if use_cuda:
        # Parse remaining args to set up inputs/animation, then render via CUDA
        ds.cli.meta(tuple(args))
        # After cli.meta returns, scene should have inputs + animation set up
        # The output should have been set via main() args
        # For CUDA: re-run with cuda_render
        print("[DepthFlow] CUDA backend requested — use scene.cuda_render() programmatically")
    else:
        ds.cli.meta(tuple(args))


def gradio(*ctx: Annotated[str, Parameter(
    allow_leading_hyphen=True,
    show=False,
)]) -> None:
    """🔵 Run depthflow's gradio webui interface"""
    from depthflow.webui import DepthGradio
    cyclopts.run(DepthGradio().launch)

def main() -> None:
    cli = App(help_flags=[])
    cli.default(scene)
    cli.command(gradio)
    cli(sys.argv[1:])

if __name__ == "__main__":
    main()
