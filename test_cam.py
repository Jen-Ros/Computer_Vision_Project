import cv2  # This is the OpenCV library

# '0' usually refers to your default webcam. 
# If you have an external camera, you might need '1'.
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

while True:
    # 1. Read a frame from the webcam
    # ret is a boolean (True/False) if the read was successful
    # frame is the actual image data
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # 2. Show the frame in a window named "My Camera"
    cv2.imshow('My Camera', frame)

    # 3. Wait for 1 millisecond for a key press
    # If the key is 'q', break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Clean up
cap.release()
cv2.destroyAllWindows()