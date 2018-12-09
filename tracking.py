import sys
import os
from os import path

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

fps = cap.get(cv2.CAP_PROP_FPS)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (35,35),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

points = []
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = np.array([[[100,263]],[[362,266]]], dtype = np.float32)

points.append(p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #plt.imshow(frame)
    #plt.show()
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #input("escriba algo: (:v)")
    # Select good points
    #print(p1)
    points.append(p1)
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

p = np.array(points)

x = p[:,0,0,0]
y = p[:,0,0,1]
t = np.arange(0,len(x)/fps,1/fps)

plt.plot(t,x, "*", label = "eje x")
plt.plot(t,y, "*", label = "eje y")
plt.legend()
plt.show()