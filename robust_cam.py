import cv2          # The Computer Vision library (OpenCV)
import numpy as np  # The Math library (handles the image arrays/matrices)
import time         # Used to force the code to wait (sleep)

print("Initializing camera...")

# 1. SETUP THE CAMERA
# '0' is usually the default webcam.
# 'cv2.CAP_DSHOW' is a specific setting for Windows to force the camera to start faster/reliably.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 2. WARM UP
# Cameras take a second to adjust exposure and focus. 
# Without this sleep, the first few frames might be black, causing the code to crash.
time.sleep(2)

# Check if the connection was successful
if not cap.isOpened():
    print("Cannot open camera")
    exit() # Stop the program immediately

print("Camera is open! Press 'q' to quit.")
empty_frame_count = 0 # Counter to track if the camera glitches

# 3. START THE MAIN LOOP
# Video is just a series of images (frames) played fast. This loop processes one image at a time.
while True:
    
    # cap.read() returns two things:
    # 'ret': A True/False boolean (Did we get a picture?)
    # 'frame': The actual image data (pixels)
    ret, frame = cap.read()

    # --- ERROR HANDLING ---
    # If the camera didn't send a picture, don't crash. Just try again.
    if not ret or frame is None:
        print("Frame empty... trying again.")
        empty_frame_count += 1
        
        # If we fail 50 times in a row, assume the camera is broken/unplugged
        if empty_frame_count > 50:
            break
        continue # Skip the rest of the code and go back to the start of the loop
    
    empty_frame_count = 0 # Reset counter if we got a good frame
    
    # 4. COLOR CONVERSION
    # We convert BGR (Blue-Green-Red) to HSV (Hue-Saturation-Value).
    # Why? In BGR, a "dark blue" is mathematically very different from "light blue".
    # In HSV, "Blue" is just a specific Hue number, making it easier to track regardless of lighting.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 5. DEFINE COLOR TARGET
    # We define the lower and upper limits of the color we want.
    # [Hue, Saturation, Value]
    lower_blue = np.array([90, 50, 50])   # Dark/Weak Blue
    upper_blue = np.array([130, 255, 255]) # Bright/Intense Blue

    # 6. CREATE THE MASK
    # This scans every pixel. Is it between lower_blue and upper_blue?
    # Yes -> Make pixel WHITE (255)
    # No  -> Make pixel BLACK (0)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 7. CLEAN UP NOISE (Morphological Operations)
    # 'Erode': Scrubs away small white specks (noise) in the background.
    mask = cv2.erode(mask, None, iterations=2)
    # 'Dilate': Expands the remaining white areas to make the object solid again.
    mask = cv2.dilate(mask, None, iterations=2)

    # 8. FIND CONTOURS
    # This looks for the "coastline" or outline of the white blobs in the mask.
    # RETR_EXTERNAL means "only give me the outer outline, ignore holes inside the object."
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 9. OBJECT DETECTION LOGIC
    if len(contours) > 0:
        # 9.1 Find the biggest blob
        # (We assume the biggest blue thing is your object, not a random speck on the wall)
        c = max(contours, key=cv2.contourArea)
        
        # 9.2 Check the size
        area = cv2.contourArea(c)
        
        # 9.3 Filter by size (Thresholding)
        # Only draw the box if the object is bigger than 1000 pixels.
        # This prevents the box from glitching on tiny background noise.
        if area > 1000:
            # Get the coordinates for a rectangle around the blob
            # x,y = top-left corner
            # w,h = width and height
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw the Green Box on the original frame
            # (0, 255, 0) is Green color code
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put text above the box
            cv2.putText(frame, "Blue Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 10. CREATE "RESULT" VIEW
    # This acts like a cookie cutter. It uses the mask to cut the blue object out of the original frame.
    # Everything not in the mask becomes black.
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 11. DISPLAY WINDOWS
    cv2.imshow('1. Real Feed', frame)  # Shows the camera with the green box
    cv2.imshow('2. Mask', mask)        # Shows the black & white computer vision view
    cv2.imshow('3. Result', result)    # Shows the isolated blue object

    # 12. EXIT LOGIC
    # Wait 1ms for a key press. If 'q' is pressed, break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 13. CLEANUP
# Release the camera so other apps (Zoom/Discord) can use it again.
cap.release()
# Close all the pop-up windows.
cv2.destroyAllWindows()


# --- EXPECTED OUTPUT EXPLANATION ---
# When you run this code, three windows will appear:
#
# Window 1: "1. Real Feed"
# - Shows your normal webcam video.
# - If you hold up a blue object large enough (>1000px area), a GREEN BOX will appear around it.
# - Text "Blue Object" will appear above the box.
#
# Window 2: "2. Mask"
# - This is a black-and-white image.
# - Your blue object will appear as a WHITE blob.
# - The background will be completely BLACK.
# - This proves the computer has successfully "separated" the object from the world.
#
# Window 3: "3. Result"
# - Shows the original colors of your object, but floating in a black void.
# - This verifies that the mask is perfectly aligned with the object.
