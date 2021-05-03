import cv2
import depthai as dai
import numpy as np
import open3d as o3d
import os, json, tempfile

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
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
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            # flip the orientation, so it looks upright, not upside-down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
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

pipeline = dai.Pipeline()

M_left = np.array([[855.849548,    0.000000,  632.435974],
                    [0.000000,  856.289001,  399.700226],
                    [0.000000,    0.000000,    1.000000]])

# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True) # The rectified streams are horizontally mirrored by default
# depth.setOutputDepth(False)
# depth.setExtendedDisparity(True)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
depth.setMedianFilter(median)

left.out.link(depth.left)
right.out.link(depth.right)

# Create depth output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

# Create left output
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
depth.rectifiedLeft.link(xout_left.input)

pcl_converter = None

with dai.Device(pipeline) as device:
  # Start pipeline
  device.startPipeline()

  # Output queue will be used to get the disparity frames from the outputs defined above
  q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

  # Output queue will be used to get the disparity frames from the outputs defined above
  q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)

  while True:

      in_left = q_left.get()
      l_frame = in_left.getFrame()
      l_frame = cv2.flip(l_frame, flipCode=1)
      cv2.imshow("left", l_frame)

      in_depth = q.get()  # blocking call, will wait until a new data has arrived
      # data is originally represented as a flat 1D array, it needs to be converted into HxW form
      depth_frame = in_depth.getFrame().astype(np.uint8)
      # print(depth_frame)
      depth_frame = np.ascontiguousarray(depth_frame)
      # frame is transformed, the color map will be applied to highlight the depth info
      # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
      # flip frame (for testing)
      # depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
      # frame is ready to be shown
      cv2.imshow("disparity", depth_frame)# Capture the key pressed
      if pcl_converter is None:
          pcl_converter = PointCloudVisualizer(M_left, 1280, 720)
      pcd = pcl_converter.rgbd_to_projection(depth_frame, l_frame, False)
      pcl_converter.visualize_pcd()
      key_pressed = cv2.waitKey(1) & 0xff

      # Stop the program if q was pressed
      if key_pressed == ord('q'):
        break