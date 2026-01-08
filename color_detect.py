import cv2
import numpy as np
import time 

print("Initializing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera is open! Press 'q' to quit.")
empty_frame_count = 0

while True:
    ret, frame = cap.read()

    # --- ERROR HANDLING (Keep this from robust_cam.py) ---
    if not ret or frame is None:
        print("Frame empty... trying again.")
        empty_frame_count += 1
        if empty_frame_count > 50:
            break
        continue
    empty_frame_count = 0

    

    #convert the image from bgr (standard) to hsv which is better for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define the range of blue color to detect
    #these numbers are specific to OpenCV's hsv scale
    lower_blue = np.array([90, 50, 50]) #lower limit of blue
    upper_blue = np.array([130, 255, 255]) #upper limit of blue

    #creating a MASK  that checks every pixel if it is within the blue range
    # if yes, make it white. If no, make it black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #BITWISE_AND, the "overlay" of the mask on the original image
    # it says: "only show the part of the image that the mask is white"
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #show all three stages
    cv2.imshow('1. Original', frame)
    cv2.imshow('2. The Mask', mask)
    cv2.imshow('3. Result', result)

    #quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()