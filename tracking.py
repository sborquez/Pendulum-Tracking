import sys
import os
from os import path

import numpy as np
import cv2 as cv

video_folder = "videos"

source = "webcam"
video_name = ""

if len(sys.argv) == 2:
    source = "video"
    video_name = sys.argv[1] 

print(f"source: {source} {path.join(video_folder,video_name)}")