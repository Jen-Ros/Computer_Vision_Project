import cv2
import numpy as np # <--- Make sure you import numpy at the top!
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
    # -----------------------------------------------------

    # === NEW COLOR DETECTION CODE STARTS HERE ===
    
    # 1. Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define Blue Range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 3. Create Mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4. Result
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 5. Show Windows (We show all 3 now)
    cv2.imshow('1. Real Feed', frame)
    cv2.imshow('2. Mask', mask)
    cv2.imshow('3. Result', result)
    
    # === NEW CODE ENDS HERE ===

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()