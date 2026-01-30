from pathlib import Path

import open3d as o3d


def visualize_sequence(point_cloud_dir: str, frame_delay_ms: int = 100):
    """
    Load and play back a sequence of PLY files in an Open3D viewer.

    Args:
        point_cloud_dir: Directory containing PLY files (frame_*.ply)
        frame_delay_ms: Delay between frames in milliseconds

    Controls:
        Space: Play/pause
        Right arrow: Next frame
        Left arrow: Previous frame
        Q/Escape: Quit
    """
    ply_dir = Path(point_cloud_dir)
    ply_files = sorted(ply_dir.glob("frame_*.ply"))

    if not ply_files:
        print(f"No PLY files found in {point_cloud_dir}")
        return

    print(f"Found {len(ply_files)} frames")
    print("Controls: Space=play/pause, Left/Right=step, Q=quit")

    clouds = [o3d.io.read_point_cloud(str(f)) for f in ply_files]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Sequence")

    current_idx = [0]
    playing = [False]

    vis.add_geometry(clouds[0])

    def update_cloud(idx):
        vis.clear_geometries()
        vis.add_geometry(clouds[idx])
        print(f"\rFrame {idx + 1}/{len(clouds)}: {ply_files[idx].name}", end="", flush=True)

    def next_frame(_vis):
        current_idx[0] = (current_idx[0] + 1) % len(clouds)
        update_cloud(current_idx[0])

    def prev_frame(_vis):
        current_idx[0] = (current_idx[0] - 1) % len(clouds)
        update_cloud(current_idx[0])

    def toggle_play(_vis):
        playing[0] = not playing[0]
        status = "Playing" if playing[0] else "Paused"
        print(f"\n{status}")

    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(263, prev_frame)  # Left arrow

    print(f"Frame 1/{len(clouds)}: {ply_files[0].name}")

    last_update = [0]
    import time

    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

        if playing[0]:
            now = time.time() * 1000
            if now - last_update[0] >= frame_delay_ms:
                current_idx[0] = (current_idx[0] + 1) % len(clouds)
                update_cloud(current_idx[0])
                last_update[0] = now

    vis.destroy_window()
    print("\nDone")
