import os
import sys
from importlib.metadata import files

import numpy as np


if sys.platform == "linux":
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    import OpenGL.GL  # Load PyOpenGL's GLX backend before selecting Genesis's Pyglet context.

    os.environ["PYOPENGL_PLATFORM"] = "pyglet"

import genesis as gs
from genesis.utils.misc import tensor_to_array


installed_files = {str(path) for path in files("genesis-world") or ()}
assert not any(path.startswith("genesis/ext/isaacgym/") for path in installed_files)
assert "genesis/ext/VolumeSampling" not in installed_files


gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="warning")

scene = gs.Scene(
    renderer=gs.renderers.Rasterizer(),
    show_viewer=False,
    profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    vis_options=gs.options.VisOptions(
        shadow=False,
        plane_reflection=False,
        segmentation_level="entity",
    ),
)

scene.add_entity(morph=gs.morphs.Plane())
box = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.3, 0.3, 0.3),
        pos=(0.0, 0.0, 0.4),
    ),
    surface=gs.surfaces.Smooth(color=(0.9, 0.1, 0.1)),
)
camera = scene.add_camera(
    res=(128, 96),
    pos=(1.2, 1.2, 0.9),
    lookat=(0.0, 0.0, 0.2),
    fov=40,
)
scene.build()
for _ in range(3):
    scene.step(update_visualizer=False)

pos = tensor_to_array(box.get_pos())
assert pos.shape[-1] == 3
assert float(pos[2]) < 0.4

rgb, _, segmentation, _ = camera.render(rgb=True, segmentation=True)
rgb = tensor_to_array(rgb)
segmentation = tensor_to_array(segmentation)
assert rgb.shape == (96, 128, 3)
assert np.isfinite(rgb).all()
assert rgb.reshape(-1, 3).std(axis=0).max() > 5.0
rgb_int = rgb.astype(np.int16)
red_pixels = (rgb_int[..., 0] - rgb_int[..., 1] > 40) & (rgb_int[..., 0] - rgb_int[..., 2] > 40)
assert np.count_nonzero(red_pixels) > 20
assert np.any(segmentation == box.idx + 1)

scene.destroy()
gs.destroy()
