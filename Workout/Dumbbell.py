import cv2
import time
import pygame
import mediapipe as mp

#Change the location path of music
mpdraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose = mppose.Pose()

cap = cv2.VideoCapture(0)
#cap.set(3, 1080)
#cap.set(4, 1020)

#counting variables
up = False
counter = 0

#Time variables
start_time = None
exercise_duration = 30

pygame.mixer.init()
pygame.mixer.music.load(r"/Music.mp3")

while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgrgb)
    # print(result.pose_landmarks)

    if start_time is None:
        start_time = time.time()

    elapsed_time = time.time() - start_time
    remaining_time = max(exercise_duration - elapsed_time, 0)

    cv2.putText(img, f"Time left: {int(remaining_time)}s", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 3)
    cv2.putText(img, f"Time left: {int(remaining_time)}s", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 2)

    if elapsed_time >= exercise_duration:
        pygame.mixer.music.stop()
        break  # Exit the loop when the timer is up

    if result.pose_landmarks:
        mpdraw.draw_landmarks(img, result.pose_landmarks, mppose.POSE_CONNECTIONS)

        points = {}
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[id] = (cx, cy)

        cv2.circle(img, points[14], 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, points[16], 15, (0, 0, 255), cv2.FILLED)
        #cv2.circle(img, points[13], 15, (255, 0, 0), cv2.FILLED)
        #cv2.circle(img, points[15], 15, (255, 0, 0), cv2.FILLED)

        if not up and points[14][1] + 5 > points[16][1]:
            up = True
            counter += 1

        elif points[14][1] < points[16][1]:
            up = False

    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

    # Coordinates for the circles
    circle1_center = (450, 600)
    circle2_center = (802, 600)
    circle3_center = (450, 650)
    circle4_center = (802, 650)
    circle_radius = 50
    circle_color = (0, 0, 0)  # Red color



    # Draw the rectangle
    cv2.rectangle(img, (450, 550), (800, 700), circle_color, thickness=cv2.FILLED)

    # Draw the circles
    cv2.circle(img, circle1_center, circle_radius, circle_color, thickness=cv2.FILLED)
    cv2.circle(img, circle2_center, circle_radius, circle_color, thickness=cv2.FILLED)
    cv2.circle(img, circle3_center, circle_radius, circle_color, thickness=cv2.FILLED)
    cv2.circle(img, circle4_center, circle_radius, circle_color, thickness=cv2.FILLED)

    cv2.putText(img, "Dumbbell Counter", (420, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 8)
    cv2.putText(img, "Dumbbell Counter", (420, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (44, 186, 246), 4)
    cv2.putText(img, "Count: " + str(counter), (445, 650), cv2.FONT_HERSHEY_DUPLEX , 2.5, (14, 94, 255), 10)

    cv2.imshow("Counter", img)
    cv2.waitKey(1)
