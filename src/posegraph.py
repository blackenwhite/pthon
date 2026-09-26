"""Slice 5: Open3D pose-graph vs raw Record3D odometry (ablation, not full SLAM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src.cloud import (
    _conf_path,
    _depth_path,
    _quat_to_R,
    _read_png,
    camera_to_world,
    scale_intrinsics,
    unproject_frame,
)
from src.ingest import Capture, Pose


def pose_to_T(pose: Pose, invert: bool = False) -> np.ndarray:
    R, t = camera_to_world(pose, invert=invert)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


def T_to_pose(base: Pose, T: np.ndarray) -> Pose:
    return Pose(
        timestamp=base.timestamp,
        frame=base.frame,
        t=T[:3, 3].copy(),
        q=_R_to_quat(T[:3, :3]),
        fx=base.fx,
        fy=base.fy,
        cx=base.cx,
        cy=base.cy,
    )


def path_length_m(Ts: np.ndarray) -> float:
    if Ts.shape[0] < 2:
        return 0.0
    d = np.diff(Ts[:, :3, 3], axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def start_end_m(Ts: np.ndarray) -> float:
    if Ts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(Ts[-1, :3, 3] - Ts[0, :3, 3]))


def _to_pcd(xyz: np.ndarray, voxel_m: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    if voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_m)
    return pcd


def _icp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init: np.ndarray,
    *,
    max_corr: float,
) -> tuple[np.ndarray, float]:
    if len(src.points) < 40 or len(tgt.points) < 40:
        return init.copy(), 0.0
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return np.asarray(reg.transformation, dtype=np.float64), float(reg.fitness)


def _rel_delta_m(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.inv(a) @ b
    return float(np.linalg.norm(d[:3, 3]))


def _information(src, tgt, trans, max_corr: float) -> np.ndarray:
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, max_corr, trans
    )
    return np.asarray(info, dtype=np.float64)


def load_camera_frame_clouds(
    cap: Capture,
    *,
    frame_stride: int,
    pixel_stride: int,
    min_confidence: int,
    invert_extrinsics: bool,
) -> tuple[list[Pose], list[np.ndarray], list[np.ndarray]]:
    """Per-frame clouds in camera coordinates, plus raw T_wc."""
    if cap.rgb_size is None:
        raise ValueError("RGB size unknown; cannot scale K to depth")
    poses = cap.poses[:: max(1, frame_stride)]
    kept_poses: list[Pose] = []
    clouds: list[np.ndarray] = []
    Ts: list[np.ndarray] = []
    depth_wh: tuple[int, int] | None = None
    I = np.eye(3, dtype=np.float64)
    z0 = np.zeros(3, dtype=np.float64)
    for pose in poses:
        depth = _read_png(_depth_path(cap.root, pose.frame))
        if depth is None or depth.ndim != 2:
            continue
        if depth_wh is None:
            depth_wh = (int(depth.shape[1]), int(depth.shape[0]))
        cpath = _conf_path(cap.root, pose.frame)
        conf = _read_png(cpath) if cpath.exists() else None
        fx, fy, cx, cy = scale_intrinsics(
            pose.fx, pose.fy, pose.cx, pose.cy, cap.rgb_size, depth_wh
        )
        pts, _cols = unproject_frame(
            depth,
            fx,
            fy,
            cx,
            cy,
            I,
            z0,
            confidence=conf,
            rgb_bgr=None,
            pixel_stride=pixel_stride,
            min_confidence=min_confidence,
        )
        if pts.shape[0] < 80:
            continue
        kept_poses.append(pose)
        clouds.append(pts.astype(np.float64))
        Ts.append(pose_to_T(pose, invert=invert_extrinsics))
    return kept_poses, clouds, Ts


@dataclass
class DriftResult:
    n_frames: int
    n_odometry_edges: int
    n_loop_attempts: int
    n_loop_edges: int
    raw_path_m: float
    opt_path_m: float
    raw_start_end_m: float
    opt_start_end_m: float
    mean_pose_delta_m: float
    mean_seq_fitness: float
    notes: list[str] = field(default_factory=list)
    raw_Ts: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 4, 4)))
    opt_Ts: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 4, 4)))
    poses_raw: list[Pose] = field(default_factory=list)
    poses_opt: list[Pose] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "n_odometry_edges": self.n_odometry_edges,
            "n_loop_attempts": self.n_loop_attempts,
            "n_loop_edges": self.n_loop_edges,
            "raw_path_m": round(self.raw_path_m, 6),
            "opt_path_m": round(self.opt_path_m, 6),
            "raw_start_end_m": round(self.raw_start_end_m, 6),
            "opt_start_end_m": round(self.opt_start_end_m, 6),
            "mean_pose_delta_m": round(self.mean_pose_delta_m, 6),
            "mean_seq_fitness": round(self.mean_seq_fitness, 6),
            "notes": list(self.notes),
            "disclaimer": (
                "Pose-graph is sequential ICP + nearby-time loop closures on a "
                "subsampled RGB-D stream. It is an ablation against raw ARKit "
                "odometry, not a claim that drift is solved."
            ),
        }


def refine_pose_graph(
    cap: Capture,
    *,
    frame_stride: int = 24,
    pixel_stride: int = 8,
    min_confidence: int = 1,
    invert_extrinsics: bool = False,
    voxel_m: float = 0.06,
    icp_corr_m: float = 0.12,
    loop_dist_m: float = 0.80,
    loop_min_sep: int = 8,
    max_loops: int = 40,
) -> DriftResult:
    notes: list[str] = []
    poses, clouds, T_list = load_camera_frame_clouds(
        cap,
        frame_stride=frame_stride,
        pixel_stride=pixel_stride,
        min_confidence=min_confidence,
        invert_extrinsics=invert_extrinsics,
    )
    n = len(clouds)
    raw = np.stack(T_list, axis=0) if n else np.zeros((0, 4, 4))
    if n < 3:
        notes.append("too few frames for a pose graph; raw odometry unchanged")
        return DriftResult(
            n_frames=n,
            n_odometry_edges=0,
            n_loop_attempts=0,
            n_loop_edges=0,
            raw_path_m=path_length_m(raw),
            opt_path_m=path_length_m(raw),
            raw_start_end_m=start_end_m(raw),
            opt_start_end_m=start_end_m(raw),
            mean_pose_delta_m=0.0,
            mean_seq_fitness=0.0,
            notes=notes,
            raw_Ts=raw,
            opt_Ts=raw.copy(),
            poses_raw=poses,
            poses_opt=list(poses),
        )

    pcds = [_to_pcd(c, voxel_m) for c in clouds]
    pg = o3d.pipelines.registration.PoseGraph()
    for T in T_list:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T.copy()))

    n_odo = 0
    n_kept_odo = 0
    fitnesses: list[float] = []
    for i in range(n - 1):
        init = np.linalg.inv(T_list[i + 1]) @ T_list[i]
        trans, fit = _icp(pcds[i], pcds[i + 1], init, max_corr=icp_corr_m)
        jump = _rel_delta_m(init, trans)
        if fit < 0.08 or jump > 0.18:
            trans = init
            n_kept_odo += 1
        info = _information(pcds[i], pcds[i + 1], trans, icp_corr_m)
        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, i + 1, trans, info, uncertain=False
            )
        )
        n_odo += 1
        fitnesses.append(fit)
    if n_kept_odo:
        notes.append(
            f"kept raw odometry on {n_kept_odo}/{n_odo} sequential edges (weak ICP or jump > 0.18 m)"
        )

    loop_attempts = 0
    n_loop = 0
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + loop_min_sep, n):
            d = float(np.linalg.norm(T_list[j][:3, 3] - T_list[i][:3, 3]))
            if d <= loop_dist_m:
                candidates.append((d, i, j))
    candidates.sort()
    for _d, i, j in candidates[:max_loops]:
        loop_attempts += 1
        init = np.linalg.inv(T_list[j]) @ T_list[i]
        trans, fit = _icp(pcds[i], pcds[j], init, max_corr=icp_corr_m)
        if fit < 0.12:
            continue
        delta = trans @ np.linalg.inv(init)
        if float(np.linalg.norm(delta[:3, 3])) > 0.40:
            continue
        info = _information(pcds[i], pcds[j], trans, icp_corr_m)
        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, j, trans, info, uncertain=True
            )
        )
        n_loop += 1

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=icp_corr_m,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    opt = np.stack([np.asarray(node.pose, dtype=np.float64) for node in pg.nodes])
    if path_length_m(opt) > 2.5 * max(path_length_m(raw), 1e-6):
        notes.append(
            "pose-graph path exploded vs raw odometry; reporting raw poses as the usable trajectory"
        )
        opt = raw.copy()
    delta = np.linalg.norm(opt[:, :3, 3] - raw[:, :3, 3], axis=1)
    poses_opt = [T_to_pose(p, T) for p, T in zip(poses, opt)]
    notes.append(
        "node poses are T_wc; edges are ICP T_cj_ci vs inv(T_j)@T_i from raw odometry"
    )
    return DriftResult(
        n_frames=n,
        n_odometry_edges=n_odo,
        n_loop_attempts=loop_attempts,
        n_loop_edges=n_loop,
        raw_path_m=path_length_m(raw),
        opt_path_m=path_length_m(opt),
        raw_start_end_m=start_end_m(raw),
        opt_start_end_m=start_end_m(opt),
        mean_pose_delta_m=float(delta.mean()) if delta.size else 0.0,
        mean_seq_fitness=float(np.mean(fitnesses)) if fitnesses else 0.0,
        notes=notes,
        raw_Ts=raw,
        opt_Ts=opt,
        poses_raw=poses,
        poses_opt=poses_opt,
    )


def write_drift_preview(result: DriftResult, out_path: Path) -> Path:
    """Top-down XZ of raw vs optimized camera translations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 720
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    if result.raw_Ts.shape[0] < 1:
        cv2.putText(img, "no poses", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        cv2.imwrite(str(out_path), img)
        return out_path
    raw = result.raw_Ts[:, [0, 2], 3]
    opt = result.opt_Ts[:, [0, 2], 3]
    pts = np.vstack([raw, opt])
    lo = pts.min(axis=0) - 0.4
    hi = pts.max(axis=0) + 0.4
    span = np.maximum(hi - lo, 0.5)

    def px(p):
        x = int((p[0] - lo[0]) / span[0] * (w - 80) + 40)
        y = int((hi[1] - p[1]) / span[1] * (h - 80) + 40)
        return x, y

    def draw(traj, color):
        pix = [px(p) for p in traj]
        for a, b in zip(pix, pix[1:]):
            cv2.line(img, a, b, color, 2)
        if pix:
            cv2.circle(img, pix[0], 6, (20, 160, 20), -1)
            cv2.circle(img, pix[-1], 6, (20, 20, 200), -1)

    draw(raw, (40, 80, 220))
    draw(opt, (40, 160, 40))
    cv2.putText(
        img,
        "blue=raw odometry  green=pose-graph  dots=start/end",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
    )
    cv2.imwrite(str(out_path), img)
    return out_path


def drift_text(result: DriftResult, *, label: str) -> str:
    d = result.to_dict()
    lines = [
        f"=== drift ablation: {label} ===",
        f"frames: {d['n_frames']}  odo_edges: {d['n_odometry_edges']}  "
        f"loop_attempts: {d['n_loop_attempts']}  loop_edges: {d['n_loop_edges']}",
        f"path_m raw/opt: {d['raw_path_m']:.3f} / {d['opt_path_m']:.3f}",
        f"start_end_m raw/opt: {d['raw_start_end_m']:.3f} / {d['opt_start_end_m']:.3f}",
        f"mean_pose_delta_m: {d['mean_pose_delta_m']:.4f}  "
        f"mean_seq_icp_fitness: {d['mean_seq_fitness']:.3f}",
    ]
    for n in result.notes[:8]:
        lines.append(f"note: {n}")
    lines.append(d["disclaimer"])
    return "\n".join(lines)
