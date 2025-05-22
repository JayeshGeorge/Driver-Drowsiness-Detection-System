from picamera2 import Picamera2
from imutils import face_utils
from threading import Thread
import numpy as np
import dlib
import cv2
import os
from gpiozero import PWMOutputDevice
import time

# Buzzer setup
buzzer = PWMOutputDevice(17)

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 30
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False

# Initialize Dlib's face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Picamera2 setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# Helper functions
def compute(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = compute(eye[1], eye[5])
    B = compute(eye[2], eye[4])
    C = compute(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return ((leftEAR + rightEAR) / 2.0, leftEye, rightEye)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

def alarm(msg):
    global alarm_status, alarm_status2, saying
    if alarm_status:
        while alarm_status:
            buzzer.frequency = 1000
            buzzer.value = 0.5
            time.sleep(0.3)
            buzzer.value = 0
            time.sleep(0.2)
    elif alarm_status2:
        saying = True
        os.system(f'espeak "{msg}"')
        saying = False

# Allow camera warm-up
for _ in range(5):
    picam2.capture_array()
    time.sleep(0.1)

print("-> Starting detection...")
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(shape[48:60])], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=alarm, args=('wake up sir',), daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                if alarm_status:
                    alarm_status = False
                    buzzer.value = 0  # Turn off buzzer

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    Thread(target=alarm, args=('yawn detected',), daemon=True).start()
            else:
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    print("-> Exiting...")
    buzzer.off()
    picam2.close()
    cv2.destroyAllWindows()
