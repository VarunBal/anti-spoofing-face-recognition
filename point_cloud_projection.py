import numpy as np
import open3d as o3d
import math

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.fx = intrinsic_matrix[0][0]
        self.fy = intrinsic_matrix[1][1]
        self.cx = intrinsic_matrix[0][2]
        self.cy = intrinsic_matrix[1][2]

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         self.fx,
                                                                         self.fy,
                                                                         self.cx,
                                                                         self.cy)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb, is_rgb):
        self.depth_map = depth_map
        self.rgb = rgb

        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)

        # TODO: query frame shape to get this, and remove the param 'is_rgb'
        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1, convert_rgb_to_intensity=False)
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1)

        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            # flip the orientation, so it looks upright, not upside-down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors

            # segment plane
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
            [a, b, c, d] = plane_model
            # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            avg_distance = calc_avg_dist(outlier_cloud.points, plane_model)
            print("average distance:", avg_distance)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        return self.pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()

M_right = np.array([[855.000122,    0.000000,  644.814514],
                    [0.000000,  855.263794,  407.305450],
                    [0.000000,    0.000000,    1.000000]])

def calc_avg_dist(points, plane):
    [a, b, c, d] = plane
    total_dist = 0
    for point in points:
        d = abs((a * point[0] + b * point[1] + c * point[2] + d))
        e = (math.sqrt(a * a + b * b + c * c))
        total_dist += d / e
    avg_dist = total_dist/len(points)
    return avg_dist