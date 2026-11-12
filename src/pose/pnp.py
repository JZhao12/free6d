"""PnP solving and the 2D-geometry fallback pose (numpy + OpenCV)."""
import cv2
import numpy as np


def _failed():
    return {
        "ok": False,
        "R": np.eye(3, dtype=np.float64),
        "t": np.zeros(3, dtype=np.float64),
        "inliers": np.zeros(0, dtype=np.int64),
    }


def solve_pnp_ransac(pts3d, pts2d, K, reproj_thr=12.0, iters=400, confidence=0.999):
    """
    EPnP + RANSAC, then Levenberg-Marquardt refinement on the inliers.

    `pts3d` (M, 3) in mm, `pts2d` (M, 2) in pixels. Returns {"ok", "R" (3, 3), "t" (3,),
    "inliers" (K,)}; ok is False (inliers empty) if the solve fails or gives fewer than 4
    inliers, non-finite values or t_z <= 0.
    """
    pts3d = np.ascontiguousarray(pts3d, dtype=np.float64).reshape(-1, 3)
    pts2d = np.ascontiguousarray(pts2d, dtype=np.float64).reshape(-1, 2)
    K = np.ascontiguousarray(K, dtype=np.float64).reshape(3, 3)
    if len(pts3d) < 4 or len(pts3d) != len(pts2d):
        return _failed()
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            K,
            None,
            iterationsCount=int(iters),
            reprojectionError=float(reproj_thr),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except Exception:
        return _failed()
    if not ok or inliers is None:
        return _failed()
    inliers = np.asarray(inliers).reshape(-1).astype(np.int64)
    if len(inliers) < 4 or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return _failed()
    try:
        rvec_r, tvec_r = cv2.solvePnPRefineLM(
            np.ascontiguousarray(pts3d[inliers]),
            np.ascontiguousarray(pts2d[inliers]),
            K,
            None,
            rvec.copy(),
            tvec.copy(),
        )
        if np.isfinite(rvec_r).all() and np.isfinite(tvec_r).all():
            rvec, tvec = rvec_r, tvec_r
    except Exception:
        pass  # keep the unrefined pose
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not (np.isfinite(R).all() and np.isfinite(t).all()) or t[2] <= 0:
        return _failed()
    return {"ok": True, "R": R, "t": t, "inliers": inliers}


def rotation_between(a, b):
    """Minimal rotation taking the direction `a` to the direction `b` (both 3-vectors)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # anti-parallel: proper rotation by 180 deg about an axis orthogonal to a
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    theta = np.arctan2(s, c)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def _crop_center_ray(K, crop):
    """Viewing ray K^-1 [u, v, 1] (z = 1) through the centre of a square crop."""
    left, top, side = crop
    u = left + side / 2.0 - 0.5  # continuous crop centre -> pixel-index coordinates
    v = top + side / 2.0 - 0.5
    return np.linalg.solve(np.asarray(K, dtype=np.float64), np.array([u, v, 1.0]))


def pose_from_template(R_t, t_t, K_t, crop_t, K_q, crop_q):
    """
    Fallback pose when PnP fails: the rotation of a template, placed from the 2D crop geometry.

    `crop_*` = (left, top, side) of the square crop in the template / query image. The
    template pose is turned by the rotation R_d between the viewing rays through the template
    and query crop centres and scaled by the ratio of the crop sides (and focal lengths), so
    the offset between the object origin and the crop centre is kept:
    R = R_d R_t, t = scale * R_d t_t.
    """
    R_t = np.asarray(R_t, dtype=np.float64).reshape(3, 3)
    t_t = np.asarray(t_t, dtype=np.float64).reshape(3)
    K_t = np.asarray(K_t, dtype=np.float64).reshape(3, 3)
    K_q = np.asarray(K_q, dtype=np.float64).reshape(3, 3)
    side_t = float(crop_t[2])
    side_q = float(crop_q[2])
    scale = (side_t / side_q) * (K_q[0, 0] / K_t[0, 0])
    d_q = _crop_center_ray(K_q, crop_q)
    d_t = _crop_center_ray(K_t, crop_t)
    R_d = rotation_between(d_t / np.linalg.norm(d_t), d_q / np.linalg.norm(d_q))
    return R_d @ R_t, scale * (R_d @ t_t)
