# import libraries
import cv2
import numpy as np
import math
import mediapipe as mp

# mediapipe initialization
mp_holistic = mp.solutions.holistic

# constants
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

dist_coeffs = np.zeros((4,1)) # Assuming no camera lens distortion

# functions
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
                                [0, 0, 1]], dtype = "double")
    return camera_matrix
    
    
# openCV implementation
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():

        ret, frame = cap.read()
        
        size = frame.shape
    
        frame = cv2.resize(frame, (640, 480))

        height, width = size[0], size[1]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = holistic.process(frame)
            
            # get face points
            face_point = face_points_coords(face_point_lm)

            # get camera parameters
            camera_matrix = cam_internal_param(size)

            # solve perspective
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, face_point, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            # get 2d projection of points
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # get point of nosetip and line projection
            p1 = ( int(face_point[0][0]), int(face_point[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # draw face points
            for p in face_point:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            # draw line indicating head orientation
            cv2.line(frame, p1, p2, (255,0,0), 2)
            
            # check head orientation
            if p2[1] > p1[1] + 50:
                cv2.putText(frame, "LOOKING DOWN", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except:
            cv2.putText(frame, "NO DETECTION", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            pass
        
        cv2.imshow('MediaPipe Pose', frame)
        
        if cv2.waitKey(5) & 0xFF == ord("q"):
          break

cap.release()
cv2.destroyAllWindows()