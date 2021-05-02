import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

# Define a source - color camera
# cam_rgb = pipeline.createColorCamera()

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
# depth.setOutputDepth(False)
depth.setExtendedDisparity(True)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
depth.setMedianFilter(median)

left.out.link(depth.left)
right.out.link(depth.right)

# Create depth output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

with dai.Device(pipeline) as device:
  # Start pipeline
  device.startPipeline()

  # Output queue will be used to get the disparity frames from the outputs defined above
  q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

  while True:

      in_depth = q.get()  # blocking call, will wait until a new data has arrived
      # data is originally represented as a flat 1D array, it needs to be converted into HxW form
      depth_frame = in_depth.getFrame().astype(np.uint8)
      # print(depth_frame)
      depth_frame = np.ascontiguousarray(depth_frame)
      # frame is transformed, the color map will be applied to highlight the depth info
      depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
      # flip frame (for testing)
      depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
      # frame is ready to be shown
      cv2.imshow("disparity", depth_frame)# Capture the key pressed
      key_pressed = cv2.waitKey(1) & 0xff

      # Stop the program if q was pressed
      if key_pressed == ord('q'):
        break