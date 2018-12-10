from time import sleep
import sys
import os
from os import path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from calculate import calculate_frec, plot, get_points

video_folder = "videos"

source = "webcam"
video_name = ""

updates = 1   #
interval = 3 # Calcular usando 1 segundo de video minimo

if len(sys.argv) == 4:
    source = "video"
    video_name = sys.argv[1] 
    interval = int(sys.argv[2])
    updates = int(sys.argv[3])

elif len(sys.argv) == 2:
    source = "video"
    video_name = sys.argv[1]

print(f"source: {source} {path.join(video_folder,video_name)}")

if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(path.join(video_folder,video_name))

iframe = -1
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = np.ceil(fps*interval)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (35,35),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
iframe += 1

points = []
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


p0 = get_points(old_gray)
#p0 = np.array([[[100,263]],[[362,266]]], dtype = np.float32)

points.append(p0)

# Create some random colors
color = np.random.randint(0,255,(p0.shape[0],3))

x0 = p0[:,0,0]
y0 = p0[:,0,1]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

min_x = old_frame.shape[0] * np.ones(x0.shape, int)
max_x = -1* np.ones(x0.shape, int)

min_y = old_frame.shape[0] * np.ones(x0.shape, int)
max_y = -1 * np.ones(x0.shape, int)

l = ox = oy = fx = fy = None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    iframe += 1
    if not ret: # if video end
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #plt.imshow(frame)
    #plt.show()

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    x1 = p1[:,0,0]
    y1 = p1[:,0,1]

    # Update x range and y range
    for i in range(x1.shape[0]):            
        if (x1[i] > max_x[i]): max_x[i] = int(x1[i])
        if (y1[i] > max_y[i]): max_y[i] = int(y1[i])
        if (x1[i] < min_x[i]): min_x[i] = int(x1[i])
        if (y1[i] < min_y[i]): min_y[i] = int(y1[i])

    #print(min_x, max_x, min_y, max_y)
    #input("escriba algo: (:v)")
    # Select good points
    #print(p1)
    points.append(p1) #iframe == len(points)

    # UPDATE VARIABLES
    if (iframe%interval_frames == 0 and updates):
        _,_ = calculate_frec(points, min_x, max_x, min_y, max_y, fps)
        updates -= 1

    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    # Draw x and y range
    for i in range(min_x.shape[0]):
        c = tuple(map(int, color[i]))
        cv2.circle(img,(min_x[i],min_y[i]),6, c,-1)
        cv2.circle(img,(max_x[i],min_y[i]),6, c,-1)
        cv2.circle(img,(int((min_x[i] + max_x[i])/2),max_y[i]),6, c,-1)
    
    # Draw pendulum center
    if ox is not None:
        cv2.circle(img, (ox, oy), 5, (255,255,255),-1)

    # Draw values
    #draw(img, x1, y1, ox, oy,  y1-y0)
    for i in range(x1.shape[0]):
        c = tuple(map(int, color[i]))
        cv2.arrowedLine(img, (x1[i], y1[i]),(int(x1[i] + 5*(x1[i]-x0[i])), int(y1[i] + 5*(y1[i]-y0[i]))), c, 4)

    # Add labels
    for i in range(x1.shape[0]):
        msg = f"[{i}] pos: ({int(x1[i])},{int(y1[i])})[px]    vel: ({int(x1[i]-x0[i])},{int(y1[i]-y0[i])})[px/frame] theta: -"
        c = tuple(map(int, color[i]))
        #print("color:", c)
        cv2.putText(img, msg, (10, 12*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c)
    cv2.putText(img, f"frame: {iframe}" , (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,125, 255))

    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    x0, y0 = x1.copy(), y1.copy()
    if source == "video":
        sleep(0.5/fps)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

plot(points, (min_x, max_x), (min_y, max_y), fps)