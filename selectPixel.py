import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

cap = cv2.VideoCapture("videos/video1.mp4")
ret, frame = cap.read()

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

