import cv2
import dlib
from scipy.spatial import distance
import time

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds and time limit
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60
TIME_LIMIT = 2  # Time limit in seconds for detection (both for eyes and mouth)

# Timer variables
eye_timer = 0
mouth_timer = 0

# Load face detector and landmark predictor
print("[INFO] Loading the predictor and starting the webcam...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes for left and right eyes and mouth
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = [shape[i] for i in LEFT_EYE]
        rightEye = [shape[i] for i in RIGHT_EYE]
        mouth = [shape[i] for i in MOUTH]

        # Calculate EAR and MAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw contours around eyes and mouth
        for (x, y) in leftEye + rightEye + mouth:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if eyes are closed for more than 2 seconds
        if avgEAR < EAR_THRESHOLD:
            eye_timer += 1  # Increment the timer
            if eye_timer >= TIME_LIMIT * 30:  # 30 FPS -> 2 seconds = 60 frames
                cv2.putText(frame, "DROWSINESS DETECTED (EYES)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            eye_timer = 0  # Reset timer when eyes are open

        # Check if mouth is open for more than 2 seconds
        if mar > MAR_THRESHOLD:
            mouth_timer += 1  # Increment the timer
            if mouth_timer >= TIME_LIMIT * 30:  # 30 FPS -> 2 seconds = 60 frames
                cv2.putText(frame, "DROWSINESS DETECTED (MOUTH)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            mouth_timer = 0  # Reset timer when mouth is closed

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
