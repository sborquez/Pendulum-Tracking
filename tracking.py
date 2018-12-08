import sys
import os
from os import path

import numpy as np
import cv2

video_folder = "videos"

source = "webcam"
video_name = ""

if len(sys.argv) == 2:
    source = "video"
    video_name = sys.argv[1] 

print(f"source: {source} {path.join(video_folder,video_name)}")

if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(path.join(video_folder,video_name))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()