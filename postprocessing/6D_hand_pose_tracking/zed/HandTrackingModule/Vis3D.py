"""Open3D viewer for the 3D hand landmarks and the camera coordinate frame."""

import open3d as o3d

NUM_LANDMARKS = 21


class Vis3D(object):
    """Minimal Open3D window showing one point cloud per detected hand.

    Args:
        window_name: Title of the Open3D window.
        frame_size: Size of the camera coordinate frame marker, in the same
            unit as the landmark coordinates (metres by default).
    """

    def __init__(self, window_name="Hand landmarks", frame_size=0.1):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        self.vis.add_geometry(self.mesh_frame)

        # Initialize left and right hand point clouds
        self.pcd_left = o3d.geometry.PointCloud()
        self.pcd_left.paint_uniform_color([1, 0.706, 0])
        self.vis.add_geometry(self.pcd_left)

        self.pcd_right = o3d.geometry.PointCloud()
        self.pcd_right.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_right)

        self.pcd_hand = o3d.geometry.PointCloud()
        self.pcd_hand.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_hand)

        self.blue = [0, 0, 1]
        self.red = [1, 0, 0]

    def show_hand(self, data, color=None):
        """Render a ``(21, 3)`` landmark array. Ignores incomplete hands."""
        if data.shape != (NUM_LANDMARKS, 3):
            return None

        self.vis.remove_geometry(self.pcd_hand)
        self.pcd_hand = o3d.geometry.PointCloud()
        self.pcd_hand.points = o3d.utility.Vector3dVector(data)
        self.pcd_hand.paint_uniform_color(color if color is not None else self.blue)
        self.vis.add_geometry(self.pcd_hand)

        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if getattr(self, "vis", None) is not None:
            self.vis.destroy_window()
            self.vis = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
