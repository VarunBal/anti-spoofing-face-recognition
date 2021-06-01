# Face authentication using keras-facenet module

# Import required modules
from keras_facenet import FaceNet
import cv2
import numpy as np
import depthai as dai
import os
import point_cloud_projection
from keras.models import load_model

filepath = "model.22-0.98.h5"
model_input_size = (64, 64)
detection_model = load_model(filepath, compile = True)

# Create a facenet object
facenet = FaceNet()
# Specify the detection confidence threshold
# Detected bounding boxes with less than 95%
# confidence will be ignored
detection_threshold = 0.95


SKIP_FRAMES = 10
RESIZE_HEIGHT = 360

# Start defining a pipeline
pipeline = dai.Pipeline()

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
depth.setOutputRectified(True)  # The rectified streams are horizontally mirrored by default
depth.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

# max_disparity = 95
#
# max_disparity *= 2 # Double the range (include if extended disparity is true)
# depth.setExtendedDisparity(True)
#
# # When we get disparity to the host, we will multiply all values with the multiplier
# # for better visualization
# multiplier = 255 / max_disparity

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
depth.setMedianFilter(median)

left.out.link(depth.left)
right.out.link(depth.right)

# Create left output
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")
depth.rectifiedRight.link(xout_right.input)

# Create depth output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

def get_embeddings(images):
  """
  This function creates a embedding
  feature vector list from the input images
  """
  # List to store feature embeddings
  embs = []
  # List for storing bounding boxes
  # detected by facenet
  bboxes = []
  # Iterate over the input images
  for image in images:
    # Get detection dictionary for each image
    detections = facenet.extract(image, detection_threshold)
    # Check if at least one face was detected
    if len(detections) > 0:
      # Add feature embedding to the list
      embs.append(detections[0]['embedding'])
      # Add detected bounding box to the list
      bboxes.append(detections[0]['box'])
    else:
      # If no face is detected by facenet
      print("No face detected.")
  # Convert feature embedding list to numpy array
  embs = np.array(embs)
  # Return feature embedding list and
  # bounding boxes list
  return embs, bboxes

# Feature embedding vector of enrolled faces
enrolled_faces = []


def enroll_face(images):
  """
  This function finds the feature embedding
  for given images and then adds it to the
  list of enrolled faces.
  This entire process is equivalent to
  face enrolment.
  """
  # Get feature embedding vector
  embeddings, _ = get_embeddings(images)
  for embedding in embeddings:
    # Add feature embedding to list of
    # enrolled faces
    enrolled_faces.append(embedding)


def delist_face(images):
  """
  This function removes a face from the list
  of enrolled faces.
  """
  # Get feature embedding vector for input images
  embeddings, _ = get_embeddings(images)
  global enrolled_faces
  if len(embeddings) > 0:
    for embedding in embeddings:
      # List of faces remaining after delisting
      remaining_faces = []
      # Iterate over the enrolled faces
      for idx, face_emb in enumerate(enrolled_faces):
        # Compute distance between feature embedding
        # for input images and the current face's
        # feature embedding
        dist = facenet.compute_distance(embedding, face_emb)
        # If the above distance is more than or equal to
        # threshold, then add the face to remaining faces list
        # Distance between feature embeddings
        # is equivalent to the difference between
        # two faces
        if dist >= authentication_threshold:
          remaining_faces.append(face_emb)
      # Update the list of enrolled faces
      enrolled_faces = remaining_faces


# The minimum distance between two faces
# to be called unique
authentication_threshold = 0.30


def authenticate_face(image):
  """
  This function checks if a face
  in the given image is present
  in the list of enrolled faces or not.
  """
  # Get feature embeddings for the input image
  embedding, bboxes = get_embeddings([image])
  # Set authenatication to False by default
  authentication = False
  # If at least one face was detected
  if len(embedding) > 0:
    # Iterate over all the enrolled faces
    for face_emb in enrolled_faces:
      # Compute the distance between the enrolled face's
      # embedding vector and the input image's
      # embedding vector
      dist = facenet.compute_distance(embedding[0],face_emb)
      # If above distance is less the threshold
      if dist < authentication_threshold:
        # Set the authenatication to True
        # meaning that the input face has been matched
        # to the current enrolled face
        authentication = True
    if authentication == True:
      # If the face was authenticated,
      # return "True" (for authentication) and the
      # bounding boxes around the detected face
      return True, bboxes[0]
    else:
      # If the face was not authenticated,
      # return "False" (for authentication) and the
      # bounding boxes around the detected face
      return False, bboxes[0]
  # Default or when no face was detected
  return None, None


def overlay_symbol(frame, img, pos=(10, 100)):
  """
  This function overlays the image of lock/unlock
  if the authentication of the input frame
  is successful/failed.
  """
  # Offset value for the image of the lock/unlock
  symbol_x_offset = pos[0]
  symbol_y_offset = pos[1]
 
  # Find top left and bottom right coordinates
  # where to place the lock/unlock image
  y1, y2 = symbol_y_offset, symbol_y_offset + img.shape[0]
  x1, x2 = symbol_x_offset, symbol_x_offset + img.shape[1]

  # Scale down alpha channel between 0 and 1
  mask = img[:, :, 3]/255.0
  # Inverse of the alpha mask
  inv_mask = 1-mask
 
  # Iterate over the 3 channels - R, G and B
  for c in range(0, 3):
    # Add the lock/unlock image to the frame
    frame[y1:y2, x1:x2, c] = (mask * img[:, :, c] +
                              inv_mask * frame[y1:y2, x1:x2, c])

# Load image of a lock in locked position
locked_img = cv2.imread(os.path.join('data', 'images', 'lock_grey.png'), -1)
# Load image of a lock in unlocked position
unlocked_img = cv2.imread(os.path.join('data', 'images', 'lock_open_grey.png'), -1)

# Frame count
count = 0

def check_if_same(roi):
  # Get a flattened 1D view of 2D numpy array
  flatten_arr = np.ravel(roi)
  # Check if all value in 2D array are equal
  result = np.all(roi == flatten_arr[0])
  print(result)


def check_in_range(roi):
  avg = np.average(roi)
  np.where(np.logical_and(roi>=avg-100, roi<=avg+100))

# wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
# wlsFilter.setLambda(8000)
# wlsFilter.setSigmaColor(1.5)

pcl_converter = None

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
  # Start pipeline
  device.startPipeline()

  # Output queue will be used to get the disparity frames from the outputs defined above
  q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)

  # Output queue will be used to get the disparity frames from the outputs defined above
  q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

  while True:

      in_right = q_right.get()
      r_frame = in_right.getFrame()
      r_frame = cv2.flip(r_frame, flipCode=1)
      # cv2.imshow("right", r_frame)

      in_depth = q.get()  # blocking call, will wait until a new data has arrived
      depth_frame = in_depth.getFrame()
      # depth_frame = (depth_frame*multiplier).astype(np.uint8)
      depth_frame = np.ascontiguousarray(depth_frame)
      depth_frame = cv2.bitwise_not(depth_frame)

      # cv2.imshow("without wls filter", cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET))
      # depth_frame = wlsFilter.filter(depth_frame, r_frame)
      # frame is transformed, the color map will be applied to highlight the depth info
      depth_frame_cmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
      # frame is ready to be shown
      cv2.imshow("disparity", depth_frame_cmap)

      # if pcl_converter is None:
      #     pcl_converter = point_cloud_projection.PointCloudVisualizer(point_cloud_projection.M_right, 1280, 720)

      # # Retrieve 'bgr' (opencv format) frame
      frame = cv2.cvtColor(r_frame,cv2.COLOR_GRAY2RGB)

      if (count % SKIP_FRAMES == 0):
        # Authenticate the face present in the frame
        authenticated, bbox = authenticate_face(frame)

      # Check if a face was detected in the frame
      if bbox:
        # If the face in the frame was authenticated
        face_roi_d = depth_frame[max(0, bbox[1]):bbox[1]+bbox[3], max(0, bbox[0]):bbox[0]+bbox[2]]
        face_roi_r = r_frame[max(0, bbox[1]):bbox[1]+bbox[3], max(0, bbox[0]):bbox[0]+bbox[2]]
        cv2.imshow("face_roi", face_roi_d)

        face_depth = np.full_like(depth_frame, np.nan, depth_frame.dtype)
        face_depth[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]] = face_roi_d
        # cv2.imshow("face_depth", face_depth)

        # pcd = pcl_converter.rgbd_to_projection(face_depth, r_frame, False)
        # pcl_converter.visualize_pcd()
        resized_face_roi_d = cv2.resize(face_roi_d, model_input_size)
        resized_face_roi_d = resized_face_roi_d/255
        resized_face_roi_d = np.expand_dims(resized_face_roi_d, axis=-1)
        resized_face_roi_d = np.expand_dims(resized_face_roi_d, axis=0)
        result = detection_model.predict(resized_face_roi_d)
        if result[0][0] > .5:
          prediction = 'spoofed'
        else:
          prediction = 'real'
        # print(result)
        print(prediction)

        # check_if_same(face_roi)

        if authenticated == True:
          # Display "Authenticated" status on the frame
          cv2.rectangle(frame, bbox, (0, 255, 0) , 2)
          cv2.putText(frame, 'Authenticated', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
          # Display lock in unlocked position
          overlay_symbol(frame, unlocked_img)
        # If the face in the frame was not authenticated
        elif authenticated == False:
          # Display "Unauthenticated" status on the frame
          cv2.rectangle(frame, bbox, (0, 0, 255), 2)
          cv2.putText(frame, 'Unauthenticated', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
          # Display lock in locked position
          overlay_symbol(frame, locked_img)
      else:
        # If no face was detected, display the same on the frame
        cv2.putText(frame, 'No Face Detected.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # Display the lock in locked position
        overlay_symbol(frame, locked_img)

      # Display instructions on the frame
      cv2.putText(frame, 'Press E to Enroll Face.', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
      cv2.putText(frame, 'Press D to Delist Face.', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
      cv2.putText(frame, 'Press Q to Quit.', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

      # Capture the key pressed
      key_pressed = cv2.waitKey(1) & 0xff

      # Enrol the face if e was pressed
      if key_pressed == ord('e'):
        enroll_face([frame])
      # Delist the face if d was pressed
      elif key_pressed == ord('d'):
        delist_face([frame])
      # Stop the program if q was pressed
      elif key_pressed == ord('q'):
        break

      # Display the final frame
      cv2.imshow("Authentication Cam", frame)

      count += 1
