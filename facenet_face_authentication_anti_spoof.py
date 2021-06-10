# Anti spoofing Face authentication

# Import required modules
import cv2
import numpy as np
import depthai as dai
import os
from keras.models import load_model
from face_auth import authenticate_face, enroll_face, delist_face

# Initial spoofed classification model
model_file = "identify-spoof.22-0.98.h5"
model_input_size = (64, 64)
detection_model = load_model(model_file, compile=True)

# Set the number of frames to skip
SKIP_FRAMES = 10

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
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7  # For depth filtering
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

# Load image of a lock in locked position
locked_img = cv2.imread(os.path.join('data', 'images', 'lock_grey.png'), -1)
# Load image of a lock in unlocked position
unlocked_img = cv2.imread(os.path.join('data', 'images', 'lock_open_grey.png'), -1)


def overlay_symbol(frame, img, pos=(65, 100)):
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


# Frame count
count = 0

# Initialize wlsFilter
# wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
# wlsFilter.setLambda(8000)
# wlsFilter.setSigmaColor(1.5)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the disparity frames from the outputs defined above
    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    # Output queue will be used to get the disparity frames from the outputs defined above
    q_depth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while True:
        # Get right camera frame
        in_right = q_right.get()
        r_frame = in_right.getFrame()
        r_frame = cv2.flip(r_frame, flipCode=1)
        # cv2.imshow("right", r_frame)

        # Get depth frame
        in_depth = q_depth.get()  # blocking call, will wait until a new data has arrived
        depth_frame = in_depth.getFrame()
        # depth_frame = (depth_frame*multiplier).astype(np.uint8)
        depth_frame = np.ascontiguousarray(depth_frame)
        depth_frame = cv2.bitwise_not(depth_frame)

        # Apply wls filter
        # cv2.imshow("without wls filter", cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET))
        # depth_frame = wlsFilter.filter(depth_frame, r_frame)

        # frame is transformed, the color map will be applied to highlight the depth info
        depth_frame_cmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        # frame is ready to be shown
        cv2.imshow("disparity", depth_frame_cmap)

        # Retrieve 'bgr' (opencv format) frame from gray scale
        frame = cv2.cvtColor(r_frame, cv2.COLOR_GRAY2RGB)

        if count % SKIP_FRAMES == 0:
            # Authenticate the face present in the frame
            authenticated, bbox = authenticate_face(frame)

        # Set default status
        status_color = (0, 0, 255)
        status = 'No Face Detected.'
        unlock = False

        # Check if a face was detected in the frame
        if bbox:
            # Get face roi from right and depth frames
            face_d = depth_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            face_r = r_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            cv2.imshow("face_roi", face_d)

            # Preprocess face depth map for classification
            resized_face_d = cv2.resize(face_d, model_input_size)
            resized_face_d = resized_face_d / 255
            resized_face_d = np.expand_dims(resized_face_d, axis=-1)
            resized_face_d = np.expand_dims(resized_face_d, axis=0)

            # Get prediction
            result = detection_model.predict(resized_face_d)
            if result[0][0] > .5:
                prediction = 'spoofed'
                is_real = False
            else:
                prediction = 'real'
                is_real = True
            print(prediction)

            # Check if face is real
            if is_real:
                # Check if the face in the frame was authenticated
                if authenticated:
                    # Set Status
                    status_color = (0, 255, 0)
                    status = 'Authenticated'
                    unlock = True
                else:
                    # Set Status
                    status = 'Unauthenticated'
            else:
                # Set Status
                status = 'Unauthenticated'
                # Display "Spoof detected" status on the bbox
                cv2.putText(frame, 'Spoof Detected', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Display bounding box
        cv2.rectangle(frame, bbox, status_color, 2)

        # Create background for showing details
        cv2.rectangle(frame, (5, 5, 175, 150), (50, 0, 0), -1)

        # Display authentication status on the frame
        cv2.putText(frame, status, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color)

        # Display lock symbol
        if unlock:
            overlay_symbol(frame, unlocked_img)
        else:
            overlay_symbol(frame, locked_img)

        # Display instructions on the frame
        cv2.putText(frame, 'Press E to Enroll Face.', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(frame, 'Press D to Delist Face.', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(frame, 'Press Q to Quit.', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        # Capture the key pressed
        key_pressed = cv2.waitKey(1) & 0xff

        # Enrol the face if e was pressed
        if key_pressed == ord('e'):
            if is_real:
                enroll_face([frame])
        # Delist the face if d was pressed
        elif key_pressed == ord('d'):
            if is_real:
                delist_face([frame])
        # Stop the program if q was pressed
        elif key_pressed == ord('q'):
            break

        # Display the final frame
        cv2.imshow("Authentication Cam", frame)

        # Increment frame count
        count += 1
