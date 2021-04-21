# Face authentication using keras-facenet module

# Import required modules
from keras_facenet import FaceNet
import cv2
import numpy as np
import depthai as dai
import os

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

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 400)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

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
left.out.link(depth.left)
right.out.link(depth.right)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Create output
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


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
  # Start pipeline
  device.startPipeline()

  # Output queue will be used to get the rgb frames from the output defined above
  q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

  # Output queue will be used to get the disparity frames from the outputs defined above
  q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

  while True:
      in_rgb = q_rgb.get()  # blocking call, will wait until a new data has arrived

      in_depth = q.get()  # blocking call, will wait until a new data has arrived
      # data is originally represented as a flat 1D array, it needs to be converted into HxW form
      depth_frame = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
      depth_frame = np.ascontiguousarray(depth_frame)
      # frame is transformed, the color map will be applied to highlight the depth info
      depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
      # frame is ready to be shown
      cv2.imshow("disparity", depth_frame)

      # # Retrieve 'bgr' (opencv format) frame
      frame = in_rgb.getCvFrame()

      if (count % SKIP_FRAMES == 0):
        # Authenticate the face present in the frame
        authenticated, bbox = authenticate_face(frame)

      # Check if a face was detected in the frame
      if bbox:
        # If the face in the frame was authenticated
        face_roi = depth_frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        cv2.imshow("face_roi", face_roi)

        check_if_same(face_roi)

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
