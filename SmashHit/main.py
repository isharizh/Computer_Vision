import cv2
import numpy as np
import HandTrack as htm
import time
import autopy

# **************
wcam, hcam = 640, 480
framered = 100
bd = 10
smooth = 2
# **************

ptime = 0
plocx, plocy = 0, 0
clocx, cloxy = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
detector = htm.handDetector(maxHands=1)
ptime = 0
wscr, hscr = autopy.screen.size()

while True:
    # Finding my hand
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Getting the tip of index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[4][1:]
        x2, y2 = lmList[8][1:]

        # Check which Fingers are Up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (framered, framered), (wcam - framered, hcam - framered), (0, 0, 0), 2)

        # Only Index Finger : Moving Mode
        if fingers[0] == 1 and fingers[1] == 1:
            # convert Coordinates
            x3 = np.interp(x1, (framered, wcam - framered), (0, wscr))
            y3 = np.interp(y1, (framered, hcam - framered), (0, hscr))

            # Smoothness
            clocx = plocx + (x3 - plocx) / smooth
            clocy = plocy + (y3 - plocy) / smooth

            # Move Mouse
            autopy.mouse.move(wscr - clocx, clocy)
            cv2.circle(img, (x1, y1), 10, (8, 227, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (8, 227, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy

            # Clicking Mode
            length, img, lineinfo = detector.findDistance(4, 8, img)
            # print(length)
            if length < 25:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 11, (2, 224, 255), cv2.FILLED)
                autopy.mouse.click()

    # Frame Rate
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    # Display
    bimg = cv2.copyMakeBorder(img, bd, bd, bd, bd, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    cv2.putText(bimg, str(int(fps)), (590, 470), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 5)
    cv2.putText(bimg, str(int(fps)), (590, 470), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(bimg, "Smash Hit Player !", (200, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3)
    # cv2.putText(bimg, "Smash Hit Player !", (200, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("Rizle", bimg)
    cv2.waitKey(1)
#Harixh