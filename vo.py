import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import os
import time

from utils import extract_features, match_features, filter_matches_distance, visualize_matches, estimate_motion, visual_odometry, Camera_Manager

height = 480
width = 640

camera = Camera_Manager(width, height)
trajactory = visual_odometry(camera, detector='orb', filter_match_distance=0.4, plot=True)

"""

# For Testing purpose 

profile = camera.pipeline.start(camera.config)
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

time.sleep(3)
img1, dep1 = camera.get_feed()
print("First image taken")
time.sleep(2)
img2, dep2 = camera.get_feed()
print("Second image taken")

kp1, des1 = extract_features(img1)
kp2, des2 = extract_features(img2)

matches = match_features(des1, des2)
print("Total Matches: {}".format(len(matches)))
matches = filter_matches_distance(matches, 0.4)
print("Filtered Matches: {}".format(len(matches)))

# image_matches = visualize_matches(img1, kp1, img2, kp2, matches)
# plt.figure(figsize=(16, 6), dpi=100)
# plt.imshow(image_matches)
# plt.show()

# rmat, tvec, _, _ = estimate_motion(matches, kp1, kp2, intr, dep1)
# print(rmat, tvec)

"""
