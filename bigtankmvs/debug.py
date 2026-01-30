"""Debug script for diagnosing MVS pipeline issues."""

import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .core import MultiViewStereoPipeline


class DebugPipeline(MultiViewStereoPipeline):
    """Instrumented pipeline that collects diagnostic data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_output_dir = self.project_dir / "debug"
        self.debug_output_dir.mkdir(exist_ok=True)

    def run_single_frame(self, frame_idx: int = 0):
        """Process a single frame with full diagnostics."""
        print(f"\n{'='*60}")
        print(f"DEBUG: Processing frame {frame_idx}")
        print(f"{'='*60}\n")

        caps = {name: cv2.VideoCapture(str(path)) for name, path in self.video_paths.items()}
        cam_names = list(caps.keys())
        pairs = list(combinations(cam_names, 2))
        proj_matrices = {name: self._get_projection_matrix(name) for name in cam_names}

        # Print calibration info
        self._print_calibration_summary()

        # Extract frames
        frames = {}
        for name, cap in caps.items():
            frame = self._extract_frame(cap, frame_idx, name)
            if frame is not None:
                frames[name] = frame
                cv2.imwrite(str(self.debug_output_dir / f"frame_{name}.jpg"), frame)

        print(f"\nExtracted {len(frames)}/{len(caps)} frames")
        print(f"Frame shape: {next(iter(frames.values())).shape}")

        # Process each pair with detailed stats
        pair_stats = []
        pair_clouds = []

        for cam1, cam2 in pairs:
            if cam1 not in frames or cam2 not in frames:
                continue

            stats = self._debug_pair(
                cam1, cam2,
                frames[cam1], frames[cam2],
                proj_matrices[cam1], proj_matrices[cam2],
                frame_idx
            )
            pair_stats.append(stats)

            if stats["points_final"] is not None and len(stats["points_final"]) > 0:
                pair_clouds.append(stats["points_final"])

        # Print summary table
        self._print_pair_summary(pair_stats)

        # Merge and analyze final cloud
        if pair_clouds:
            merged = self._merge_point_clouds(pair_clouds)
            print(f"\n{'='*60}")
            print("MERGED POINT CLOUD")
            print(f"{'='*60}")
            print(f"Total points: {len(merged)}")
            if len(merged) > 0:
                print(f"X range: [{merged[:, 0].min():.1f}, {merged[:, 0].max():.1f}] mm")
                print(f"Y range: [{merged[:, 1].min():.1f}, {merged[:, 1].max():.1f}] mm")
                print(f"Z range: [{merged[:, 2].min():.1f}, {merged[:, 2].max():.1f}] mm")

            # Save combined visualization with cameras
            self._save_debug_scene(merged, cam_names)
        else:
            print("\nNo points reconstructed!")

        for cap in caps.values():
            cap.release()

    def _print_calibration_summary(self):
        """Print camera calibration summary."""
        print("CAMERA CALIBRATION")
        print("-" * 60)
        print(f"{'Camera':<10} {'Focal':<10} {'Translation (mm)':<30}")
        print("-" * 60)

        for name, cam in self.cameras.items():
            fx = cam["matrix"][0, 0]
            t = cam["translation"]
            print(f"{name:<10} {fx:<10.1f} [{t[0]:>8.1f}, {t[1]:>8.1f}, {t[2]:>8.1f}]")

    def _debug_pair(self, cam1, cam2, img1, img2, P1, P2, frame_idx):
        """Process a camera pair with full diagnostics."""
        stats = {
            "pair": f"{cam1}-{cam2}",
            "matches_raw": 0,
            "matches_filtered": 0,
            "reproj_err_mean": None,
            "reproj_err_max": None,
            "points_before_filter": 0,
            "points_after_filter": 0,
            "points_final": None,
            "xyz_range": None,
        }

        # Feature matching (uses parent class method which handles all matcher types)
        pts1, pts2 = self._match_features(img1, img2, cam1, cam2)
        stats["matches_filtered"] = len(pts1)

        # Save match visualization
        self._save_match_visualization(img1, img2, pts1, pts2, cam1, cam2, frame_idx)

        if len(pts1) < 1:
            return stats

        # Triangulation with detailed error analysis
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        stats["points_before_filter"] = len(pts3d)

        # Compute reprojection errors
        pts3d_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])

        proj1 = (P1 @ pts3d_h.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        err1 = np.linalg.norm(proj1 - pts1, axis=1)

        proj2 = (P2 @ pts3d_h.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        err2 = np.linalg.norm(proj2 - pts2, axis=1)

        max_err = np.maximum(err1, err2)
        stats["reproj_err_mean"] = float(np.mean(max_err))
        stats["reproj_err_max"] = float(np.max(max_err))

        # Filter by reprojection error
        good_mask = max_err < self.reprojection_error_threshold
        pts3d_filtered = pts3d[good_mask]
        stats["points_after_filter"] = len(pts3d_filtered)

        if len(pts3d_filtered) > 0:
            stats["points_final"] = pts3d_filtered
            stats["xyz_range"] = (
                (pts3d_filtered[:, 0].min(), pts3d_filtered[:, 0].max()),
                (pts3d_filtered[:, 1].min(), pts3d_filtered[:, 1].max()),
                (pts3d_filtered[:, 2].min(), pts3d_filtered[:, 2].max()),
            )

        return stats

    def _save_match_visualization(self, img1, img2, pts1, pts2, cam1, cam2, frame_idx):
        """Save feature match visualization."""
        # Convert to keypoints
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in pts1]
        kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in pts2]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

        # Draw matches (subsample if too many)
        max_draw = 100
        if len(matches) > max_draw:
            idx = np.linspace(0, len(matches) - 1, max_draw, dtype=int)
            matches = [matches[i] for i in idx]

        vis = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        path = self.debug_output_dir / f"matches_{cam1}_{cam2}.jpg"
        cv2.imwrite(str(path), vis)

    def _print_pair_summary(self, pair_stats):
        """Print summary table of pair statistics."""
        print(f"\n{'='*60}")
        print("PAIR-WISE STATISTICS")
        print(f"{'='*60}")
        print(f"{'Pair':<15} {'Raw':<8} {'Filt':<8} {'Tri':<8} {'Good':<8} {'Err':<10}")
        print("-" * 60)

        for s in pair_stats:
            err_str = f"{s['reproj_err_mean']:.2f}" if s['reproj_err_mean'] else "-"
            print(f"{s['pair']:<15} {s['matches_raw']:<8} {s['matches_filtered']:<8} "
                  f"{s['points_before_filter']:<8} {s['points_after_filter']:<8} {err_str:<10}")

        print("-" * 60)
        total_good = sum(s['points_after_filter'] for s in pair_stats)
        print(f"Total good points: {total_good}")

        # Print 3D ranges for pairs that produced points
        print(f"\n{'='*60}")
        print("3D COORDINATE RANGES (mm)")
        print(f"{'='*60}")
        print(f"{'Pair':<15} {'X range':<25} {'Y range':<25} {'Z range':<25}")
        print("-" * 90)

        for s in pair_stats:
            if s['xyz_range']:
                x, y, z = s['xyz_range']
                print(f"{s['pair']:<15} [{x[0]:>8.1f}, {x[1]:>8.1f}] "
                      f"[{y[0]:>8.1f}, {y[1]:>8.1f}] [{z[0]:>8.1f}, {z[1]:>8.1f}]")

    def _save_debug_scene(self, points, cam_names):
        """Save point cloud with camera frustums for 3D visualization."""
        geometries = []

        # Point cloud
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(pcd)

        # Camera frustums
        colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0.5, 0], [0.5, 0, 1]
        ]

        for i, name in enumerate(cam_names):
            cam = self.cameras[name]
            rvec = cam["rotation"]
            tvec = cam["translation"]
            R, _ = cv2.Rodrigues(rvec)

            # Camera center in world coordinates: C = -R^T @ t
            cam_center = -R.T @ tvec

            # Create a small coordinate frame at camera position
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
            frame.rotate(R.T, center=[0, 0, 0])
            frame.translate(cam_center)
            geometries.append(frame)

            # Create frustum lines
            frustum = self._create_frustum(cam, R, cam_center, colors[i % len(colors)])
            geometries.append(frustum)

        # World origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        geometries.append(origin)

        # Save combined scene
        combined = o3d.geometry.PointCloud()
        for g in geometries:
            if isinstance(g, o3d.geometry.PointCloud):
                combined += g

        path = self.debug_output_dir / "debug_scene.ply"
        o3d.io.write_point_cloud(str(path), combined)
        print(f"\nSaved debug scene to {path}")

        # Also save as individual elements for Open3D viewer
        o3d.io.write_triangle_mesh(
            str(self.debug_output_dir / "world_origin.ply"), origin
        )

        print(f"Camera centers (world coords, mm):")
        for name in cam_names:
            cam = self.cameras[name]
            R, _ = cv2.Rodrigues(cam["rotation"])
            C = -R.T @ cam["translation"]
            print(f"  {name}: [{C[0]:>8.1f}, {C[1]:>8.1f}, {C[2]:>8.1f}]")

    def _create_frustum(self, cam, R, center, color, scale=100):
        """Create a line set representing a camera frustum."""
        K = cam["matrix"]
        w, h = cam["size"]

        # Image corners in normalized camera coords
        corners_2d = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float64)

        # Unproject to camera frame (z=1)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        corners_cam = np.zeros((4, 3))
        for i, (u, v) in enumerate(corners_2d):
            corners_cam[i] = [(u - cx) / fx, (v - cy) / fy, 1.0]

        # Scale and transform to world
        corners_cam *= scale
        corners_world = (R.T @ corners_cam.T).T + center

        # Create line set
        points = [center.tolist()] + corners_world.tolist()
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Rectangle
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        return line_set


def run_debug(project_name: str, frame_idx: int = 0, **kwargs):
    """Run debug pipeline on a single frame."""
    pipeline = DebugPipeline(project_name, **kwargs)
    pipeline.run_single_frame(frame_idx)
    print(f"\nDebug outputs saved to: {pipeline.debug_output_dir}")


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "calibration_121525"
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_debug(project, frame)
