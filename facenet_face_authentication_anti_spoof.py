# Anti spoofing Face authentication

# Import required modules
import cv2
import numpy as np
import depthai as dai
import os
import time
from keras.models import load_model
from face_auth import authenticate_face, enroll_face, delist_face, authenticate_emb


def create_depthai_pipeline():
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
    depth.setExtendedDisparity(True)  # For better close range depth perception

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

    xin_frame = pipeline.createXLinkIn()
    nn_out = pipeline.createXLinkOut()

    xin_frame.setStreamName("inFrame")
    nn_out.setStreamName("nn")

    nn_path = "data/depth-classification-models/depth_classification_ipscaled_model.blob"
    # Define sources and outputs
    nn = pipeline.createNeuralNetwork()

    # Properties
    nn.setBlobPath(nn_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Linking
    xin_frame.out.link(nn.input)
    nn.out.link(nn_out.input)

    arcface_in_frame = pipeline.createXLinkIn()
    arcface_in_frame.setStreamName("arc_in")

    face_rec_nn = pipeline.createNeuralNetwork()
    face_rec_nn.setBlobPath("data/face-rec-model/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
    # arcface_in_frame.out.link(face_rec_nn.input)

    arc_out = pipeline.createXLinkOut()
    arc_out.setStreamName('arc_out')
    face_rec_nn.out.link(arc_out.input)

    det_in_frame = pipeline.createXLinkIn()
    det_in_frame.setStreamName("det_in")

    face_det_nn = pipeline.createMobileNetDetectionNetwork()
    face_det_nn.setConfidenceThreshold(0.75)
    face_det_nn.setBlobPath("data/face-det-model/face-detection-adas-0001.blob")
    # det_in_frame.out.link(face_det_nn.input)

    det_out = pipeline.createXLinkOut()
    det_out.setStreamName('det_out')
    face_det_nn.out.link(det_out.input)

    face_det_manip = pipeline.createImageManip()
    # face_det_manip.initialConfig.setHorizontalFlip(True)
    face_det_manip.initialConfig.setResize(672, 384)
    face_det_manip.initialConfig.setKeepAspectRatio(False)
    # face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    # det_in_frame.out.link(face_det_manip.inputImage)
    # depth.rectifiedRight.link(face_det_manip.inputImage)
    face_det_manip.out.link(face_det_nn.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'rec_manip' to crop the initial frame
    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)
    script.setScriptPath("script.py")

    copy_manip = pipeline.createImageManip()
    depth.rectifiedRight.link(copy_manip.inputImage)
    # copy_manip.initialConfig.setHorizontalFlip(True)
    copy_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    # copy_manip.setNumFramesPool(20)

    copy_manip.out.link(face_det_manip.inputImage)
    copy_manip.out.link(script.inputs['frame'])

    face_det_nn.out.link(script.inputs['face_det_in'])
    # We are only interested in timestamp, so we can sync depth frames with NN output
    # face_det_nn.passthrough.link(script.inputs['face_pass'])

    face_rec_manip = pipeline.createImageManip()

    script.outputs['manip_cfg'].link(face_rec_manip.inputConfig)
    script.outputs['manip_img'].link(face_rec_manip.inputImage)

    face_rec_manip.out.link(face_rec_nn.input)

    return pipeline


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


# Initial spoofed classification model
# MODEL_FILE = "identify-spoof_with_ext.23-1.00.h5"
MODEL_INPUT_SIZE = (64, 64)
# detection_model = load_model(MODEL_FILE, compile=True)
DET_INPUT_SIZE = (672, 384)
REC_INPUT_SIZE = (112, 112)

# def verify_face(depth_frame, bbox):
#     # Get face roi from right and depth frames
#     face_d = depth_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
#     cv2.imshow("face_roi", face_d)
#
#     # Preprocess face depth map for classification
#     resized_face_d = cv2.resize(face_d, MODEL_INPUT_SIZE)
#     resized_face_d = resized_face_d / 255
#     resized_face_d = np.expand_dims(resized_face_d, axis=-1)
#     resized_face_d = np.expand_dims(resized_face_d, axis=0)
#
#     # Get prediction
#     result = detection_model.predict(resized_face_d)
#     if result[0][0] > .5:
#         prediction = 'spoofed'
#         is_real = False
#     else:
#         prediction = 'real'
#         is_real = True
#     print(prediction)
#
#     return is_real


def display_info(frame, bbox, status, status_color, fps):
    # Display bounding box
    cv2.rectangle(frame, bbox, status_color[status], 2)

    # If spoof detected
    if status == 'Spoof Detected':
        # Display "Spoof detected" status on the bbox
        cv2.putText(frame, "Spoofed", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Create background for showing details
    cv2.rectangle(frame, (5, 5, 175, 150), (50, 0, 0), -1)

    # Display authentication status on the frame
    cv2.putText(frame, status, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Display lock symbol
    if status == 'Authenticated':
        overlay_symbol(frame, unlocked_img)
    else:
        overlay_symbol(frame, locked_img)

    # Display instructions on the frame
    cv2.putText(frame, 'Press E to Enroll Face.', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(frame, 'Press D to Delist Face.', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(frame, 'Press Q to Quit.', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))


# Frame count
frame_count = 0

# Placeholder fps value
fps = 0

# Set the number of frames to skip for authentication
SKIP_FRAMES = 10

# Used to record the time when we processed last frames
prev_frame_time = 0

# Used to record the time at which we processed current frames
new_frame_time = 0

# Set status colors
status_color = {
    'Authenticated': (0, 255, 0),
    'Unauthenticated': (0, 0, 255),
    'Spoof Detected': (0, 0, 255),
    'No Face Detected': (0, 0, 255)
}

pipeline = create_depthai_pipeline()

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the right camera frames from the outputs defined above
    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    # Output queue will be used to get the disparity frames from the outputs defined above
    q_depth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    # Input queue will be used to send video frames to the device.
    q_in = device.getInputQueue(name="inFrame")

    # Output queue will be used to get nn data from the video frames.
    q_clas = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Input queue will be used to send video frames to the device.
    # q_arc_in = device.getInputQueue(name="arc_in")

    # Output queue will be used to get nn data from the video frames.
    q_rec = device.getOutputQueue(name="arc_out", maxSize=4, blocking=False)

    # Input queue will be used to send video frames to the device.
    # q_det_in = device.getInputQueue(name="det_in")

    # Output queue will be used to get nn data from the video frames.
    q_det = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)

    while True:
        # Get right camera frame
        in_right = q_right.get()
        r_frame = in_right.getFrame()
        # r_frame = cv2.flip(r_frame, flipCode=1)

        # Get depth frame
        in_depth = q_depth.get()  # blocking call, will wait until a new data has arrived
        depth_frame = in_depth.getFrame()
        depth_frame = cv2.flip(depth_frame, flipCode=1)
        depth_frame = np.ascontiguousarray(depth_frame)
        depth_frame = cv2.bitwise_not(depth_frame)

        # frame is transformed, the color map will be applied to highlight the depth info
        depth_frame_cmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        # frame is ready to be shown
        cv2.imshow("disparity", depth_frame_cmap)

        # Retrieve 'bgr' (opencv format) frame from gray scale
        frame = cv2.cvtColor(r_frame, cv2.COLOR_GRAY2BGR)

        if frame_count % SKIP_FRAMES == 0:
            # Authenticate the face present in the frame
            # _, bbox = authenticate_face(frame)
            bbox = None

            # resized_frame = cv2.resize(frame, DET_INPUT_SIZE).transpose(2, 0, 1)
            #
            # frame_img = dai.ImgFrame()
            # frame_img.setFrame(resized_frame)
            # frame_img.setWidth(DET_INPUT_SIZE[0])
            # frame_img.setHeight(DET_INPUT_SIZE[1])

            # q_det_in.send(frame_img)
            inDet = q_det.tryGet()
            if inDet is not None:
                detections = inDet.detections
                # for detection in detections:
                if len(detections) is not 0:
                    detection = detections[0]
                    # print(detection.confidence)
                    x = int(detection.xmin*DET_INPUT_SIZE[0])
                    y = int(detection.ymin*DET_INPUT_SIZE[1])
                    w = int(detection.xmax*DET_INPUT_SIZE[0]-detection.xmin*DET_INPUT_SIZE[0])
                    h = int(detection.ymax*DET_INPUT_SIZE[1]-detection.ymin*DET_INPUT_SIZE[1])
                    bbox = (x, y, w, h)

        face_embedding = None
        authenticated = False

        # Check if a face was detected in the frame
        if bbox:

            # face = frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            # # Preprocess face for rec
            # resized_face = cv2.resize(face, REC_INPUT_SIZE)
            # face_img = dai.ImgFrame()
            # face_img.setFrame(resized_face)
            # face_img.setWidth(REC_INPUT_SIZE[0])
            # face_img.setHeight(REC_INPUT_SIZE[1])

            # q_arc_in.send(face_img)
            inRec = q_rec.tryGet()
            if inRec is not None:
                face_embedding = inRec.getFirstLayerFp16()
                # print(len(face_embedding))
                authenticated = authenticate_emb(face_embedding)

            # Check if face is real or spoofed
            # Get face roi from right and depth frames
            face_d = depth_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            cv2.imshow("face_roi", face_d)

            # Preprocess face depth map for classification
            resized_face_d = cv2.resize(face_d, MODEL_INPUT_SIZE)
            # resized_face_d = resized_face_d / 255
            resized_face_d = resized_face_d.astype('float16')

            img = dai.ImgFrame()
            img.setFrame(resized_face_d)
            img.setWidth(MODEL_INPUT_SIZE[0])
            img.setHeight(MODEL_INPUT_SIZE[1])
            img.setType(dai.ImgFrame.Type.GRAYF16)

            q_in.send(img)

            inClas = q_clas.tryGet()

            is_real = None
            if inClas is not None:
                # Get prediction

                cnn_output = inClas.getLayerFp16("dense_2/Sigmoid")
                # print(cnn_output)
                if cnn_output[0] > .5:
                    prediction = 'spoofed'
                    is_real = False
                else:
                    prediction = 'real'
                    is_real = True
                print(prediction)

            # is_real = verify_face(depth_frame, bbox)

            if is_real:
                # Check if the face in the frame was authenticated
                if authenticated:
                    # Authenticated
                    status = 'Authenticated'
                else:
                    # Unauthenticated
                    status = 'Unauthenticated'
            else:
                # Spoof detected
                status = 'Spoof Detected'
        else:
            # No face detected
            status = 'No Face Detected'

        # Display info on frame
        display_info(frame, bbox, status, status_color, fps)

        # Calculate average fps
        if frame_count % SKIP_FRAMES == 0:
            # Time when we finish processing last 100 frames
            new_frame_time = time.time()

            # Fps will be number of frame processed in one second
            fps = 1 / ((new_frame_time - prev_frame_time)/SKIP_FRAMES)
            prev_frame_time = new_frame_time

        # Capture the key pressed
        key_pressed = cv2.waitKey(1) & 0xff

        # Enrol the face if e was pressed
        if key_pressed == ord('e'):
            if is_real:
                enroll_face([face_embedding])
        # Delist the face if d was pressed
        elif key_pressed == ord('d'):
            if is_real:
                delist_face([face_embedding])
        # Stop the program if q was pressed
        elif key_pressed == ord('q'):
            break

        # Display the final frame
        cv2.imshow("Authentication Cam", frame)

        # Increment frame count
        frame_count += 1

cv2.destroyAllWindows()
