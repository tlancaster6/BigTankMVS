"""Benchmark script comparing feature matchers on a single frame."""

import csv
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .core import SUPPORTED_MATCHERS, MultiViewStereoPipeline


class BenchmarkResult:
    """Container for benchmark results from a single matcher."""

    def __init__(self, matcher_name: str):
        self.matcher_name = matcher_name
        self.total_matches = 0
        self.total_triangulated = 0
        self.total_after_filter = 0
        self.reproj_errors = []
        self.processing_time = 0.0
        self.points = None

    @property
    def mean_reproj_error(self) -> float | None:
        if not self.reproj_errors:
            return None
        return float(np.mean(self.reproj_errors))

    @property
    def max_reproj_error(self) -> float | None:
        if not self.reproj_errors:
            return None
        return float(np.max(self.reproj_errors))

    @property
    def point_count(self) -> int:
        return len(self.points) if self.points is not None else 0

    def bbox(self) -> tuple | None:
        if self.points is None or len(self.points) == 0:
            return None
        return (
            (self.points[:, 0].min(), self.points[:, 0].max()),
            (self.points[:, 1].min(), self.points[:, 1].max()),
            (self.points[:, 2].min(), self.points[:, 2].max()),
        )


def run_benchmark(
    project_name: str,
    frame_idx: int = 0,
    data_root: str = "data",
    matchers: tuple = SUPPORTED_MATCHERS,
    **pipeline_kwargs,
) -> list[BenchmarkResult]:
    """Run benchmark comparing matchers on a single frame.

    Args:
        project_name: Project directory name
        frame_idx: Frame index to process
        data_root: Root data directory
        matchers: Tuple of matcher names to benchmark
        **pipeline_kwargs: Additional arguments passed to MultiViewStereoPipeline

    Returns:
        List of BenchmarkResult objects
    """
    project_dir = Path(data_root) / project_name
    output_dir = project_dir / "benchmark"
    output_dir.mkdir(exist_ok=True)

    results = []

    for matcher in matchers:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {matcher}")
        print(f"{'='*60}")

        result = BenchmarkResult(matcher)

        # Initialize pipeline (model loading not timed)
        pipeline = MultiViewStereoPipeline(
            project_name=project_name,
            data_root=data_root,
            matcher=matcher,
            **pipeline_kwargs,
        )

        # Open video captures
        caps = {name: cv2.VideoCapture(str(path)) for name, path in pipeline.video_paths.items()}
        cam_names = list(caps.keys())
        pairs = list(combinations(cam_names, 2))
        proj_matrices = {name: pipeline._get_projection_matrix(name) for name in cam_names}

        # Extract frames (not timed - same for all matchers)
        frames = {}
        for name, cap in caps.items():
            frame = pipeline._extract_frame(cap, frame_idx, name)
            if frame is not None:
                frames[name] = frame

        if len(frames) < 2:
            print(f"Insufficient frames extracted ({len(frames)}), skipping matcher")
            for cap in caps.values():
                cap.release()
            continue

        # Benchmark matching and triangulation
        pair_clouds = []
        start_time = time.perf_counter()

        for cam1, cam2 in pairs:
            if cam1 not in frames or cam2 not in frames:
                continue

            # Match features
            pts1, pts2 = pipeline._match_features(frames[cam1], frames[cam2], cam1, cam2)
            result.total_matches += len(pts1)

            if len(pts1) == 0:
                continue

            # Triangulate
            P1, P2 = proj_matrices[cam1], proj_matrices[cam2]
            pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            result.total_triangulated += len(pts3d)

            # Compute reprojection errors
            pts3d_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])

            proj1 = (P1 @ pts3d_h.T).T
            proj1 = proj1[:, :2] / proj1[:, 2:3]
            err1 = np.linalg.norm(proj1 - pts1, axis=1)

            proj2 = (P2 @ pts3d_h.T).T
            proj2 = proj2[:, :2] / proj2[:, 2:3]
            err2 = np.linalg.norm(proj2 - pts2, axis=1)

            max_err = np.maximum(err1, err2)
            result.reproj_errors.extend(max_err.tolist())

            # Filter by reprojection error
            good_mask = max_err < pipeline.reprojection_error_threshold
            pts3d_filtered = pts3d[good_mask]
            result.total_after_filter += len(pts3d_filtered)

            if len(pts3d_filtered) > 0:
                pair_clouds.append(pts3d_filtered)

        result.processing_time = time.perf_counter() - start_time

        # Merge point clouds
        if pair_clouds:
            result.points = pipeline._merge_point_clouds(pair_clouds)

        # Release captures
        for cap in caps.values():
            cap.release()

        # Save point cloud
        if result.points is not None and len(result.points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(result.points)
            ply_path = output_dir / f"{matcher}.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"Saved {len(result.points)} points to {ply_path.name}")

        # Print summary
        print(f"  Matches: {result.total_matches}")
        print(f"  Triangulated: {result.total_triangulated}")
        print(f"  After filter: {result.total_after_filter}")
        print(f"  Final points: {result.point_count}")
        if result.mean_reproj_error is not None:
            print(f"  Mean reproj error: {result.mean_reproj_error:.3f} px")
            print(f"  Max reproj error: {result.max_reproj_error:.3f} px")
        print(f"  Processing time: {result.processing_time:.3f} s")

        results.append(result)

    # Write CSV
    csv_path = output_dir / "benchmark_results.csv"
    _write_csv(results, csv_path)
    print(f"\n{'='*60}")
    print(f"Results saved to {csv_path}")

    return results


def _write_csv(results: list[BenchmarkResult], path: Path):
    """Write benchmark results to CSV."""
    fieldnames = [
        "matcher",
        "total_matches",
        "total_triangulated",
        "total_after_filter",
        "final_points",
        "mean_reproj_error",
        "max_reproj_error",
        "processing_time_s",
        "bbox_x_min",
        "bbox_x_max",
        "bbox_y_min",
        "bbox_y_max",
        "bbox_z_min",
        "bbox_z_max",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            bbox = r.bbox()
            row = {
                "matcher": r.matcher_name,
                "total_matches": r.total_matches,
                "total_triangulated": r.total_triangulated,
                "total_after_filter": r.total_after_filter,
                "final_points": r.point_count,
                "mean_reproj_error": f"{r.mean_reproj_error:.4f}" if r.mean_reproj_error else "",
                "max_reproj_error": f"{r.max_reproj_error:.4f}" if r.max_reproj_error else "",
                "processing_time_s": f"{r.processing_time:.4f}",
                "bbox_x_min": f"{bbox[0][0]:.2f}" if bbox else "",
                "bbox_x_max": f"{bbox[0][1]:.2f}" if bbox else "",
                "bbox_y_min": f"{bbox[1][0]:.2f}" if bbox else "",
                "bbox_y_max": f"{bbox[1][1]:.2f}" if bbox else "",
                "bbox_z_min": f"{bbox[2][0]:.2f}" if bbox else "",
                "bbox_z_max": f"{bbox[2][1]:.2f}" if bbox else "",
            }
            writer.writerow(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark feature matchers")
    parser.add_argument("project", help="Project directory name")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to process")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument(
        "--matchers",
        nargs="+",
        default=list(SUPPORTED_MATCHERS),
        choices=SUPPORTED_MATCHERS,
        help="Matchers to benchmark",
    )

    args = parser.parse_args()

    run_benchmark(
        project_name=args.project,
        frame_idx=args.frame,
        data_root=args.data_root,
        matchers=tuple(args.matchers),
    )
