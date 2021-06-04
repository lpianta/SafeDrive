# import libraries
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# mediapipe initialization
mp_holistic = mp.solutions.holistic

# constants
left_eye_lm = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
right_eye_lm = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

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
    ear_threshold = ear_threshold[~np.isnan(ear_threshold)] # ~ = is not
    # get the mean
    ear_threshold = np.mean(np.array(ear_threshold))
    return ear_threshold

# openCV implementation
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        frame = cv2.resize(frame, (640, 480))

        height, width, _ = frame.shape
        
        # Convert the BGR image to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # get results from mediapipe
            results = holistic.process(frame)
            
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

            # Draw convex hull
            cv2.drawContours(frame, [leye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [reye_hull], -1, (0, 255, 0), 1)

            # take some time to initialize values
            if frame_counter < 30:
                cv2.putText(frame, "CALIBRATING", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
                ear_threshold.append(ear)
            
            # after initializing the values calculate the right threshold
            else:
                ear_threshold = get_ear_thresh(ear_threshold)

                # print EAR on the frame
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

                # check if EAR goes below the threshold for a number of frames
                if ear < ear_threshold:
                    ear_frame_counter += 1

                    if ear_frame_counter >= 10:
                        cv2.putText(frame, "WARNING!", (20, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0))

                else:
                    ear_frame_counter = 0
        except:
            cv2.putText(frame, "NO DETECTION", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            pass

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()