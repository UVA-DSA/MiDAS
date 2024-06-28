# import the opencv library 
import cv2 
import numpy as np

# define video capture objects for both cameras
vid1 = cv2.VideoCapture(2)
vid2 = cv2.VideoCapture(3)

while True: 
    # Capture the video frame from both cameras
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    if ret1 and ret2:
        # Resize frames to fit them side by side if necessary
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # Concatenate the frames horizontally
        combined_frame = np.hstack((frame1, frame2))

        # Display the resulting frame
        cv2.imshow('Split Window Display', combined_frame)
        
        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop, release the capture objects
vid1.release()
vid2.release()
# Destroy all the windows
cv2.destroyAllWindows()
