# import libraries
import argparse
import cv2
import math
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread

# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--model_config", type=str, default="./models/ssd_mobilenet_v3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
                help="path to model config file")
ap.add_argument("-w", "--model_weights", type=str, default="./models/ssd_mobilenet_v3/frozen_inference_graph.pb",
                help="path to model weights file")
ap.add_argument("-n", "--names", type=str, default="./models/ssd_mobilenet_v3/coco.names",
                help="path to categories name file")
ap.add_argument("-i", "--input", type=int, default=1,
                help="index of webcam on system")
args = vars(ap.parse_args())

# mediapipe initialization
mp_holistic = mp.solutions.holistic

# mobilenet model setting
class_name = []
class_file = args["names"]
with open(class_file, 'rt') as f:
    class_name = f.read().rstrip("\n").split("\n")

config = args["model_config"]
weights = args["model_weights"]

net = cv2.dnn_DetectionModel(weights, config)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(False)

# constants
left_eye_lm = [33, 246, 161, 160, 159, 158, 157,
               173, 133, 155, 154, 153, 145, 144, 163, 7]

right_eye_lm = [263, 466, 388, 387, 386, 385, 384,
                398, 362, 382, 381, 380, 374, 373, 390, 249]

face_point_lm = [1,     # nose
                 152,   # chin
                 130,   # left eye corner
                 359,   # right eye corner
                 61,    # mouth left corner
                 291]   # mouth right corner

model_points = np.array([
    (0.0, 0.0, 0.0),             # nose tip
    (0.0, -330.0, -65.0),        # chin
    (-225.0, 170.0, -135.0),     # left eye corner
    (225.0, 170.0, -135.0),      # right eye corner
    (-150.0, -150.0, -125.0),    # mouth left corner
    (150.0, -150.0, -125.0)      # mouth right corner
])

dist_coeffs = np.zeros((4, 1))  # Assuming no camera lens distortion

# global variables
frame_counter = 0
ear_frame_counter = 0
ear_threshold = []

# functions


def eye_coords(frame, landmarks):
    height, width, _ = frame.shape
    eye_x = []
    eye_y = []
    for lm in landmarks:
        # mp return normalized coords, so we multiply
        eye_x.append(int((results.face_landmarks.landmark[lm].x) * width))
        eye_y.append(int((results.face_landmarks.landmark[lm].y) * height))
    eye = list(zip(eye_x, eye_y))
    eye = np.array(eye, dtype="int")
    return eye


def eye_aspect_ratio(eye):
    # euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[3], eye[13])
    B = dist.euclidean(eye[5], eye[11])
    # euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[8])
    # eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def get_ear_thresh(ear_threshold):
    # convert ear_threshold to numpy array
    ear_threshold = np.array(ear_threshold)
    # remove NaN values from the array
    ear_threshold = ear_threshold[~np.isnan(ear_threshold)]  # ~ = is not
    # get the mean
    ear_threshold = np.mean(np.array(ear_threshold))
    return ear_threshold


def face_points_coords(face_point_lm):
    face_point_x = []
    face_point_y = []
    for lm in face_point_lm:
        # mp return normalized coords, so we multiply
        face_point_x.append(results.face_landmarks.landmark[lm].x * width)
        face_point_y.append(results.face_landmarks.landmark[lm].y * height)
    face_point = list(zip(face_point_x, face_point_y))
    return np.array(face_point)


def cam_internal_param(frame_size):
    focal_length = frame_size[1]
    center = (frame_size[1]/2, frame_size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    return camera_matrix


# openCV implementation
cap = cv2.VideoCapture(args["input"])

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        frame = cv2.resize(frame, (640, 480))

        size = frame.shape

        height, width = size[0], size[1]

        # Convert the BGR image to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # get results from mediapipe
            results = holistic.process(frame)

            # get detection from mobilenet
            class_ids, conf, bbox = net.detect(frame, confThreshold=0.35)

            # update frame counter
            if results.face_landmarks:
                frame_counter += 1

            # get eyes coords
            left_eye = eye_coords(frame, left_eye_lm)
            right_eye = eye_coords(frame, right_eye_lm)

            # get EARs
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # average of both EARs
            ear = (left_ear + right_ear) / 2

            # find convex hull for the eyes
            leye_hull = cv2.convexHull(left_eye)
            reye_hull = cv2.convexHull(right_eye)

            # get face points
            face_point = face_points_coords(face_point_lm)

            # get camera parameters
            camera_matrix = cam_internal_param(size)

            # solve perspective
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, face_point, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            # get 2d projection of points
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array(
                [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # get point of nosetip and line projection
            p1 = (int(face_point[0][0]), int(face_point[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]),
                  int(nose_end_point2D[0][0][1]))

            # check head orientation
            if p2[1] > p1[1] + 50:
                cv2.putText(frame, "LOOKING DOWN", (20, 200),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 4)

            # take some time to initialize EARs
            if frame_counter < 30:
                cv2.putText(frame, "CALIBRATING", (20, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
                ear_threshold.append(ear)

            # after initializing the values calculate the right threshold
            else:
                ear_threshold = get_ear_thresh(ear_threshold)

                # print EAR on the frame
                cv2.putText(
                    frame, f"EAR: {round(ear, 2)}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

                # check if EAR goes below the threshold for a number of frames
                if ear < ear_threshold:
                    ear_frame_counter += 1

                    if ear_frame_counter >= 10:

                        cv2.putText(frame, "WARNING!", (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)

                else:
                    ear_frame_counter = 0

            # draw boxes and prediction if there is cellphone detection
            if len(class_ids) != 0:
                for class_id, confidence, box in zip(class_ids.flatten(), conf.flatten(), bbox):
                    if class_id == 77:
                        cv2.rectangle(frame, box, (0, 255, 0), 2)
                        cv2.putText(frame, class_name[class_id - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            # box_list = list(zip(class_ids.flatten(), conf.flatten(), bbox))
            # if box_list[0][0] == 77:
            #     phone_box = box_list[0][2]
            #     cv2.rectangle(frame, phone_box, (0, 255, 0), 2)
            #     cv2.putText(frame, "CELL PHONE", (phone_box[0] + 10, phone_box[1] - 10),
            #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            # draw convex hull on eyes
            cv2.drawContours(frame, [leye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [reye_hull], -1, (0, 255, 0), 1)

            # draw face points
            for p in face_point:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # draw line indicating head orientation
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        except:
            cv2.putText(frame, "NO DETECTION", (20, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            pass

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
