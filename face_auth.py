from keras_facenet import FaceNet
import numpy as np

# Create a facenet object
facenet = FaceNet()

# Specify the detection confidence threshold
# Detected bounding boxes with less than 95%
# confidence will be ignored
detection_threshold = 0.95

# Feature embedding vector of enrolled faces
enrolled_faces = []

# The minimum distance between two faces
# to be called unique
authentication_threshold = 0.30


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
            dist = facenet.compute_distance(embedding[0], face_emb)
            # If above distance is less the threshold
            if dist < authentication_threshold:
                # Set the authenatication to True
                # meaning that the input face has been matched
                # to the current enrolled face
                authentication = True
        if authentication:
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
