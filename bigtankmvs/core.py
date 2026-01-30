import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import toml
import torch
from kornia.feature import DISK, DeDoDe, LightGlueMatcher, LoFTR, laf_from_center_scale_ori
from tqdm import tqdm

SUPPORTED_MATCHERS = ("loftr", "disk_lightglue", "dedode_lightglue")


class MultiViewStereoPipeline:
    """Frame-by-frame multi-view stereo reconstruction pipeline."""

    def __init__(
        self,
        project_name: str,
        data_root: str = "data",
        frame_step: int = 1800,
        downsample_factor: float = 0.5,
        matcher: str = "loftr",
        loftr_confidence_threshold: float = 0.5,
        lightglue_max_keypoints: int = 2048,
        lightglue_detection_threshold: float = 0.0,
        lightglue_filter_threshold: float = 0.1,
        reprojection_error_threshold: float = 2.0,
        duplicate_distance_threshold: float = 5.0,
        video_extensions: tuple = ("mp4", "avi"),
        cam_regex: str = r"([^-]+)",
    ):
        """
        Args:
            project_name: Directory name within data_root containing videos/ and calibration_reoriented.toml
            data_root: Root data directory
            frame_step: Process every nth frame
            downsample_factor: Spatial downsampling factor for frames
            matcher: Feature matcher to use ("loftr", "disk_lightglue", "dedode_lightglue")
            loftr_confidence_threshold: Minimum confidence for LoFTR matches
            lightglue_max_keypoints: Maximum keypoints for LightGlue detectors (DISK/DeDoDe)
            lightglue_detection_threshold: Keypoint detection threshold for LightGlue detectors
            lightglue_filter_threshold: Match confidence threshold for LightGlue
            reprojection_error_threshold: Maximum reprojection error in pixels
            duplicate_distance_threshold: Distance threshold for merging duplicate points (mm)
            video_extensions: Video file extensions to search for
            cam_regex: Regex to extract camera name from video filename
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available")

        if matcher not in SUPPORTED_MATCHERS:
            raise ValueError(f"Unsupported matcher '{matcher}'. Choose from: {SUPPORTED_MATCHERS}")

        self.device = torch.device("cuda")

        self.project_name = project_name
        self.data_root = Path(data_root)
        self.project_dir = self.data_root / project_name
        self.frame_step = frame_step
        self.downsample_factor = downsample_factor
        self.matcher = matcher
        self.loftr_confidence_threshold = loftr_confidence_threshold
        self.lightglue_filter_threshold = lightglue_filter_threshold
        self.reprojection_error_threshold = reprojection_error_threshold
        self.duplicate_distance_threshold = duplicate_distance_threshold
        self.video_extensions = video_extensions
        self.cam_regex = cam_regex

        self.cameras = self._load_calibration()
        self.video_paths = self._discover_videos()

        print(f"Loaded {len(self.cameras)} cameras, found {len(self.video_paths)} videos")

        self._init_matcher(lightglue_max_keypoints, lightglue_detection_threshold)

    def _load_calibration(self):
        calib_path = self.project_dir / "calibration_reoriented.toml"
        calib_data = toml.load(calib_path)

        cameras = {}
        for key, cam in calib_data.items():
            if key == "metadata":
                continue
            name = cam["name"]
            cameras[name] = {
                "size": tuple(cam["size"]),
                "matrix": np.array(cam["matrix"], dtype=np.float64),
                "distortions": np.array(cam["distortions"], dtype=np.float64),
                "rotation": np.array(cam["rotation"], dtype=np.float64),
                "translation": np.array(cam["translation"], dtype=np.float64),
            }
        return cameras

    def _discover_videos(self):
        videos_dir = self.project_dir / "videos"
        video_paths = {}

        for ext in self.video_extensions:
            for path in videos_dir.glob(f"*.{ext}"):
                match = re.search(self.cam_regex, path.stem)
                if match:
                    cam_name = match.group(1)
                    if cam_name in self.cameras:
                        video_paths[cam_name] = path

        return video_paths

    def _init_matcher(self, max_keypoints, detection_threshold):
        """Initialize the feature matcher based on self.matcher setting."""
        # Store detection params for use in forward pass
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold

        if self.matcher == "loftr":
            self.loftr = LoFTR(pretrained="outdoor").to(self.device).eval()
        elif self.matcher == "disk_lightglue":
            self.detector = DISK.from_pretrained("depth").to(self.device).eval()
            self.lightglue = LightGlueMatcher("disk").to(self.device).eval()
        elif self.matcher == "dedode_lightglue":
            self.detector = DeDoDe.from_pretrained().to(self.device).eval()
            self.lightglue = LightGlueMatcher("dedodeg").to(self.device).eval()

    def _compute_roll_angle(self, cam_name):
        """Compute roll angle for rectification based on camera X-axis orientation.

        For downward-looking cameras, aligning based on world-Z is unstable since
        world-Z is nearly perpendicular to the image plane. Instead, we measure
        the angle of the camera's X-axis projected onto the world XY plane.

        Returns the angle in radians to rotate the image so that the world X-axis
        aligns with image-right across all cameras.
        """
        rvec = self.cameras[cam_name]["rotation"]
        R, _ = cv2.Rodrigues(rvec)

        # R transforms world->camera, so R.T transforms camera->world
        # Camera X-axis in world coordinates is the first column of R.T
        cam_x_world = R.T[:, 0]

        # Angle of camera X-axis from world +X axis (projected onto XY plane)
        angle = np.arctan2(cam_x_world[1], cam_x_world[0])

        return angle

    def _rectify_image_pair(self, img1, img2, cam1_name, cam2_name):
        """Rotate both images to align world X-axis with image-right.

        This ensures consistent orientation across all cameras, improving
        feature matching for downward-looking camera arrays.

        Returns:
            rect_img1: Rectified first image
            rect_img2: Rectified second image
            M1: Affine transform matrix for img1
            M2: Affine transform matrix for img2
        """
        K1 = self._get_scaled_intrinsics(cam1_name)
        K2 = self._get_scaled_intrinsics(cam2_name)

        angle1 = self._compute_roll_angle(cam1_name)
        angle2 = self._compute_roll_angle(cam2_name)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Rotate around optical center (principal point)
        M1 = cv2.getRotationMatrix2D((K1[0, 2], K1[1, 2]), np.degrees(angle1), 1.0)
        M2 = cv2.getRotationMatrix2D((K2[0, 2], K2[1, 2]), np.degrees(angle2), 1.0)

        rect_img1 = cv2.warpAffine(img1, M1, (w1, h1))
        rect_img2 = cv2.warpAffine(img2, M2, (w2, h2))

        return rect_img1, rect_img2, M1, M2

    def _unrectify_keypoints(self, kpts, M):
        """Transform keypoints from rectified coordinates back to original.

        Args:
            kpts: Nx2 array of keypoint coordinates in rectified image
            M: 2x3 affine transform matrix used for rectification

        Returns:
            Nx2 array of keypoint coordinates in original image
        """
        if len(kpts) == 0:
            return kpts

        M_inv = cv2.invertAffineTransform(M)
        ones = np.ones((len(kpts), 1))
        kpts_h = np.hstack([kpts, ones])
        kpts_orig = (M_inv @ kpts_h.T).T

        return kpts_orig

    def _get_scaled_intrinsics(self, cam_name):
        K = self.cameras[cam_name]["matrix"].copy()
        K[0, 0] *= self.downsample_factor
        K[1, 1] *= self.downsample_factor
        K[0, 2] *= self.downsample_factor
        K[1, 2] *= self.downsample_factor
        return K

    def _get_projection_matrix(self, cam_name):
        K = self._get_scaled_intrinsics(cam_name)
        rvec = self.cameras[cam_name]["rotation"]
        tvec = self.cameras[cam_name]["translation"]
        R, _ = cv2.Rodrigues(rvec)
        P = K @ np.hstack([R, tvec.reshape(3, 1)])
        return P

    def _extract_frame(self, cap, frame_idx, cam_name):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None

        if self.downsample_factor != 1.0:
            new_size = (
                int(frame.shape[1] * self.downsample_factor),
                int(frame.shape[0] * self.downsample_factor),
            )
            frame = cv2.resize(frame, new_size)

        K = self._get_scaled_intrinsics(cam_name)
        dist = self.cameras[cam_name]["distortions"]
        frame = cv2.undistort(frame, K, dist)

        return frame

    def _match_features(self, img1, img2, cam1_name=None, cam2_name=None):
        """Match features between two images.

        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            cam1_name: Camera name for img1 (required for LoFTR rectification)
            cam2_name: Camera name for img2 (required for LoFTR rectification)

        Returns:
            Tuple of (kpts1, kpts2) - Nx2 arrays of matched keypoint coordinates
        """
        if self.matcher == "loftr":
            return self._match_loftr(img1, img2, cam1_name, cam2_name)
        else:
            return self._match_lightglue(img1, img2)

    def _match_loftr(self, img1, img2, cam1_name, cam2_name):
        """Match features using LoFTR with up-vector rectification."""
        # Rectify images to align up-vectors
        rect_img1, rect_img2, M1, M2 = self._rectify_image_pair(img1, img2, cam1_name, cam2_name)

        gray1 = cv2.cvtColor(rect_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rect_img2, cv2.COLOR_BGR2GRAY)

        t1 = torch.from_numpy(gray1).float()[None, None] / 255.0
        t2 = torch.from_numpy(gray2).float()[None, None] / 255.0
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)

        with torch.no_grad():
            result = self.loftr({"image0": t1, "image1": t2})

        kpts0 = result["keypoints0"].cpu().numpy()
        kpts1 = result["keypoints1"].cpu().numpy()
        confidence = result["confidence"].cpu().numpy()

        mask = confidence >= self.loftr_confidence_threshold

        # Transform keypoints back to original image coordinates
        kpts0_orig = self._unrectify_keypoints(kpts0[mask], M1)
        kpts1_orig = self._unrectify_keypoints(kpts1[mask], M2)

        return kpts0_orig, kpts1_orig

    def _match_lightglue(self, img1, img2):
        """Match features using a detector (DISK/DeDoDe) + LightGlueMatcher."""
        # Convert to RGB tensors normalized to [0, 1]
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        t1 = torch.from_numpy(rgb1).float().permute(2, 0, 1)[None] / 255.0
        t2 = torch.from_numpy(rgb2).float().permute(2, 0, 1)[None] / 255.0
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)

        hw1 = torch.tensor(t1.shape[2:], device=self.device)
        hw2 = torch.tensor(t2.shape[2:], device=self.device)

        with torch.no_grad():
            # Detect keypoints and descriptors
            if self.matcher == "disk_lightglue":
                kpts1, descs1 = self._detect_disk(t1)
                kpts2, descs2 = self._detect_disk(t2)
            else:  # dedode_lightglue
                kpts1, descs1 = self._detect_dedode(t1)
                kpts2, descs2 = self._detect_dedode(t2)

            # Convert keypoints to LAFs (Local Affine Frames)
            lafs1 = laf_from_center_scale_ori(kpts1[None], torch.ones(1, len(kpts1), 1, 1, device=self.device))
            lafs2 = laf_from_center_scale_ori(kpts2[None], torch.ones(1, len(kpts2), 1, 1, device=self.device))

            # Match with LightGlueMatcher
            dists, idxs = self.lightglue(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

        # Extract matched keypoints
        kpts1_np = kpts1.cpu().numpy()
        kpts2_np = kpts2.cpu().numpy()
        idxs_np = idxs.cpu().numpy()

        return kpts1_np[idxs_np[:, 0]], kpts2_np[idxs_np[:, 1]]

    def _detect_disk(self, img_tensor):
        """Detect features using DISK, returning keypoints and descriptors."""
        features_list = self.detector(
            img_tensor,
            n=self.max_keypoints,
            pad_if_not_divisible=True,
        )
        feats = features_list[0]
        return feats.keypoints, feats.descriptors

    def _detect_dedode(self, img_tensor):
        """Detect features using DeDoDe, returning keypoints and descriptors."""
        keypoints, scores, descriptors = self.detector(
            img_tensor,
            n=self.max_keypoints,
            apply_imagenet_normalization=True,
        )
        # DeDoDe returns batched tensors, take first image
        return keypoints[0], descriptors[0]

    def _triangulate_pair(self, pts1, pts2, P1, P2):
        if len(pts1) < 1:
            return np.empty((0, 3))

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        pts3d_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])

        proj1 = (P1 @ pts3d_h.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        err1 = np.linalg.norm(proj1 - pts1, axis=1)

        proj2 = (P2 @ pts3d_h.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        err2 = np.linalg.norm(proj2 - pts2, axis=1)

        max_err = np.maximum(err1, err2)
        mask = max_err < self.reprojection_error_threshold

        return pts3d[mask]

    def _merge_point_clouds(self, clouds):
        if not clouds:
            return np.empty((0, 3))

        merged = np.vstack(clouds)
        if len(merged) == 0:
            return merged

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd = pcd.voxel_down_sample(voxel_size=self.duplicate_distance_threshold)

        return np.asarray(pcd.points)

    def _save_point_cloud(self, points, frame_idx):
        output_dir = self.project_dir / "point_clouds"
        output_dir.mkdir(exist_ok=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # TODO: Add RGB colors

        output_path = output_dir / f"frame_{frame_idx:06d}.ply"
        o3d.io.write_point_cloud(str(output_path), pcd)
        return output_path

    def process(self):
        caps = {name: cv2.VideoCapture(str(path)) for name, path in self.video_paths.items()}

        first_cap = next(iter(caps.values()))
        total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cam_names = list(caps.keys())
        pairs = list(combinations(cam_names, 2))

        proj_matrices = {name: self._get_projection_matrix(name) for name in cam_names}

        n_frames = len(range(0, total_frames, self.frame_step))
        print(f"Processing {n_frames} frames from {len(cam_names)} cameras ({len(pairs)} pairs)")

        for frame_idx in tqdm(range(0, total_frames, self.frame_step), desc="Frames"):
            frames = {}
            for name, cap in caps.items():
                frame = self._extract_frame(cap, frame_idx, name)
                if frame is not None:
                    frames[name] = frame

            if len(frames) < 2:
                print(f"Skipping frame {frame_idx}: insufficient cameras")
                continue

            pair_clouds = []
            for cam1, cam2 in tqdm(pairs, desc="Pairs", leave=False):
                if cam1 not in frames or cam2 not in frames:
                    continue

                pts1, pts2 = self._match_features(frames[cam1], frames[cam2], cam1, cam2)

                if len(pts1) == 0:
                    continue

                cloud = self._triangulate_pair(
                    pts1, pts2, proj_matrices[cam1], proj_matrices[cam2]
                )

                if len(cloud) > 0:
                    pair_clouds.append(cloud)

            merged = self._merge_point_clouds(pair_clouds)
            if len(merged) > 0:
                path = self._save_point_cloud(merged, frame_idx)
                print(f"Frame {frame_idx}: {len(merged)} points -> {path.name}")
            else:
                print(f"Frame {frame_idx}: no points reconstructed")

        for cap in caps.values():
            cap.release()

        print("Processing complete")
