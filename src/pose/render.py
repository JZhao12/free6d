"""
pyrender renderer that reproduces the scene setup of src/poses/pyrender.py (the one that
produced the segmentation templates) but for arbitrary poses and intrinsics.

Poses are T_m2c (mm, OpenCV camera); K follows the pixel-index convention. Unlike
src/poses/pyrender.py, this renderer is single-sampled (no MSAA) so that the renderings are
reproducible.
"""
import logging
import os
import os.path as osp

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh

from src.utils.trimesh_utils import as_mesh

# size of the segmentation templates rendered by src/poses/pyrender.py
SEG_TEMPLATE_HW = (480, 640)


def seg_template_K(dataset_name):
    """
    Intrinsics (pixel-index convention) of the segmentation templates. pyrender takes
    cx / cy as continuous coordinates, hence the -0.5.
    """
    if "tless" in dataset_name:
        fx, fy, cx, cy = 1075.65091572, 1073.90347929, 360.0, 270.0
    else:
        fx, fy, cx, cy = 572.4114, 573.57043, 325.2611, 242.04899
    return np.array(
        [[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rotation_from_z_to(direction):
    """Smallest rotation taking (0, 0, 1) to the unit vector `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = float(np.dot(z, direction))
    if c < -1.0 + 1e-9:  # opposite direction: half turn around x
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


_MSAA_DISABLED = False


def disable_msaa():
    """
    Make pyrender's offscreen framebuffer single-sampled (process-wide, idempotent).

    pyrender hard-codes 4x MSAA; the multisampled colour image is not bit-reproducible on
    all drivers (+-1 grey level in a few pixels from call to call), which is enough to
    change mutual matches and RANSAC inlier sets. With 0 samples colour and depth are
    deterministic and the depth is the one of the pixel centre.
    """
    global _MSAA_DISABLED
    if _MSAA_DISABLED:
        return
    import pyrender.renderer as pyrender_renderer

    original = pyrender_renderer.glRenderbufferStorageMultisample

    def single_sampled(target, samples, internalformat, width, height):
        return original(target, 0, internalformat, width, height)

    pyrender_renderer.glRenderbufferStorageMultisample = single_sampled
    _MSAA_DISABLED = True


class ObjectRenderer:
    """
    One pyrender.Scene (camera, spot light and one mesh node per object) and one
    pyrender.OffscreenRenderer for the whole renderer. A pyrender mesh can only be bound to a
    single GL context, and a renderer re-uploads meshes whenever the rendered scene changes, so
    scenes / renderers must not be duplicated per object or per image size. Only the requested
    object is visible in a given render call.
    """

    def __init__(self, cad_dir, dataset_name):
        disable_msaa()
        self.cad_dir = cad_dir
        self.dataset_name = dataset_name
        self._is_tless = "tless" in dataset_name
        self._meshes = {}  # obj_id -> (pyrender mesh, info)
        self._scene = None  # the single shared pyrender.Scene (created lazily)
        self._camera = None
        self._light = None
        self._light_node = None
        self._nodes = {}  # obj_id -> mesh node in the shared scene
        self._visible_id = None  # obj_id of the object currently visible in the scene
        self._renderer = None  # the single OffscreenRenderer (created at the first render)
        self._renderer_hw = None  # (H, W) of the current viewport
        # camera frame of pyrender (OpenGL) vs OpenCV, as in src/poses/pyrender.py
        self._cam_pose = np.diag([1.0, -1.0, -1.0, 1.0])

    def _load(self, obj_id):
        if obj_id in self._meshes:
            return self._meshes[obj_id]
        path = osp.join(self.cad_dir, f"obj_{obj_id:06d}.ply")
        mesh = trimesh.load_mesh(path)
        centroid = np.asarray(mesh.bounding_box.centroid, dtype=np.float64).copy()
        extents = np.asarray(mesh.extents, dtype=np.float64).copy()
        info = {"centroid": centroid, "extents": extents}
        mesh.apply_scale(0.001)  # mm -> m
        if self._is_tless:
            # uniform colors, as for the templates
            mesh.visual.face_colors = np.ones((len(mesh.faces), 3)) * 0.4
            mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 3)) * 0.4
            py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            py_mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh))
        self._meshes[obj_id] = (py_mesh, info)
        return self._meshes[obj_id]

    def mesh_info(self, obj_id):
        return dict(self._load(obj_id)[1])

    def _ensure_scene(self):
        """Create the shared scene with the camera and the spot light (no mesh yet)."""
        if self._scene is not None:
            return self._scene
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([0.02, 0.02, 0.02, 1.0]),
        )
        self._camera = pyrender.IntrinsicsCamera(
            fx=1.0, fy=1.0, cx=0.0, cy=0.0, znear=0.05, zfar=100000
        )
        scene.add(self._camera, pose=self._cam_pose)
        self._light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=0.6,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        self._light_node = scene.add(self._light, pose=self._cam_pose)
        self._scene = scene
        return self._scene

    def _ensure_node(self, obj_id):
        """Add the mesh of `obj_id` to the shared scene (invisible) the first time it is requested."""
        if obj_id in self._nodes:
            return self._nodes[obj_id]
        py_mesh, _ = self._load(obj_id)
        scene = self._ensure_scene()
        py_mesh.is_visible = False
        self._nodes[obj_id] = scene.add(
            py_mesh, pose=np.eye(4), name=f"obj_{obj_id:06d}"
        )
        return self._nodes[obj_id]

    def _show_only(self, obj_id):
        """Make `obj_id` the only visible object of the shared scene."""
        if self._visible_id == obj_id:
            return
        if self._visible_id is not None:
            self._meshes[self._visible_id][0].is_visible = False
        self._meshes[obj_id][0].is_visible = True
        self._visible_id = obj_id

    def _offscreen(self, hw):
        """The single OffscreenRenderer, created at the first call and resized afterwards."""
        H, W = int(hw[0]), int(hw[1])
        if self._renderer is None:
            self._renderer = pyrender.OffscreenRenderer(W, H)
        elif self._renderer_hw != (H, W):
            # pyrender resizes its framebuffer on the next render
            self._renderer.viewport_width = W
            self._renderer.viewport_height = H
        self._renderer_hw = (H, W)
        return self._renderer

    def render(self, obj_id, T_m2c, K, hw):
        """
        Render the object at T_m2c (4x4, t in mm) with intrinsics K (3x3) on an (H, W) image.
        Returns rgb uint8 (H, W, 3) and depth_mm float32 (H, W) (0 = background).
        """
        T_m2c = np.asarray(T_m2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        mesh_node = self._ensure_node(obj_id)
        scene = self._scene
        self._show_only(obj_id)
        camera = self._camera
        camera.fx, camera.fy = float(K[0, 0]), float(K[1, 1])
        camera.cx, camera.cy = float(K[0, 2]) + 0.5, float(K[1, 2]) + 0.5

        t_m = T_m2c[:3, 3] / 1000.0
        pose = np.eye(4)
        pose[:3, :3] = T_m2c[:3, :3]
        pose[:3, 3] = t_m
        scene.set_pose(mesh_node, pose)

        # spot light at the camera origin, pointing at the object (local -Z = t / |t|)
        dist = float(np.linalg.norm(t_m))
        light_pose = np.eye(4)
        light_pose[:3, :3] = _rotation_from_z_to(t_m / dist) @ np.diag([1.0, -1.0, -1.0])
        scene.set_pose(self._light_node, light_pose)
        self._light.intensity = 0.6 * (dist / 0.4) ** 2  # keep the brightness of 0.4 m

        rgba, depth = self._offscreen(hw).render(
            scene, pyrender.constants.RenderFlags.RGBA
        )
        rgb = np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)
        return rgb, (depth * 1000.0).astype(np.float32)

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception as e:  # the GL context may already be gone
                logging.warning(f"Failed to delete an OffscreenRenderer: {e}")
        self._renderer = None
        self._renderer_hw = None
        self._scene = None
        self._camera = None
        self._light = None
        self._light_node = None
        self._nodes.clear()
        self._visible_id = None
        self._meshes.clear()
