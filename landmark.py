import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture
cap = cv2.VideoCapture(0)


def get_eye_coordinates(landmarks, eye_points):
    points = [landmarks.part(point) for point in eye_points]
    return np.array([(point.x, point.y) for point in points], dtype=np.int32)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]

        left_eye = get_eye_coordinates(landmarks, left_eye_points)
        right_eye = get_eye_coordinates(landmarks, right_eye_points)

        # Draw the eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()