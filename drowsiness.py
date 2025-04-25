import numpy as np
import cv2
import time
import dlib
from imutils import face_utils
import pygame
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()
account_sid= os.getenv("twilio_accound_sid")
auth_token= os.getenv("twilio_auth_token")

#initialize sound mixer
pygame.mixer.init()
alert_sound="alarm.wav"
sound_count=0

#initilazing sleep timer
sleep_start_time= None
sms_sent=False

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def playsound():
    if not pygame.mixer.music.get_busy():  # Check if sound is already playing
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play(-1)  # Loop sound continuously

def compute_dist(a, b):
    return np.linalg.norm(a - b)

def eyeblink(a, b, c, d, e, f):
    vert = compute_dist(b, d) + compute_dist(c, e)  # Vertical eye distance
    hori = compute_dist(a, f)  # Horizontal eye distance
    ear = vert / (2.0 * hori)

    if ear > 0.25:
        return 2  # Open eyes
    elif 0.18 <= ear <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping





def send_sms_alert():
    
    client=Client(account_sid,auth_token)
    message=client.messages.create(
        to=os.getenv("to"),
        from_=os.getenv("from"),
        body="WARNING!! Driver is sleeping")
    
while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect(gray)

    left_blink, right_blink = -1, -1

    if len(faces) > 0 :
        for face in faces:
            
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            
            landmarks = predict(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            if landmarks.shape[0] != 68:
                print("Error: Incorrect number of facial landmarks detected")
                continue

            left_blink = eyeblink(landmarks[36], landmarks[37],
                                    landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = eyeblink(landmarks[42], landmarks[43], landmarks[44],
                                    landmarks[47], landmarks[46], landmarks[45])

    if left_blink != -1 and right_blink != -1:
        if left_blink == 0 and right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep == 16:
                status = "SLEEPING!!"
                playsound()
                color = (255, 0, 0)
                if sleep_start_time is None:
                    sleep_start_time=time.time()
                
                if time.time()- sleep_start_time >= 3 and not sms_sent:
                    print("An alert message is sent to emergency contact")
                    send_sms_alert()
                    sms_sent=True
        elif left_blink == 1 and right_blink == 1:
            drowsy += 1
            sleep = 0
            active = 0
            if drowsy > 10:
                status = "Drowsy......might sleep"
                color = (0, 0, 255)
        else:
            active += 1
            drowsy = 0
            sleep = 0
            if active > 6:
                status = "ACTIVE"
                color = (0, 255, 0)
                sleep_start_time=None
                sms_sent=False
                pygame.mixer.music.stop()
                sound_count=0
    else:
        status="No face detected"
        color= (0,0,0)
        sound_count=0

    cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 3)

    if faces:
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break
cap.release()
cv2.destroyAllWindows()

