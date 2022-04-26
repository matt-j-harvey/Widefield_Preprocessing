import numpy as np
import tables
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
from scipy.io import loadmat
from scipy.ndimage.filters import uniform_filter1d
from scipy import ndimage
from skimage.measure import find_contours
import cv2

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def load_video_as_numpy_array(video_file):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Frames: ", frameCount)

    even_frame_intensities = []
    odd_frame_intensities = []

    # Extract Selected Frames
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        frame_index += 1
        frame_value = np.mean(frame)

        if frame_index % 2 == 0:
            even_frame_intensities.append(frame_value)
        else:
            odd_frame_intensities.append(frame_value)

    cap.release()

    return frameCount, even_frame_intensities, odd_frame_intensities




# Load AI Recroder Data
ai_file_location = r"/media/matthew/29D46574463D2856/Rig_2_mousecam_Test/AIrecorder_newfile.mat"
ai_contents = loadmat(ai_file_location)
ai_contents = ai_contents["AIrecorder_file"][0][0][11]
print(np.shape(ai_contents))


# Get Mousecam Trigger Onsets
mousecam_trace = ai_contents[:, 13]
threshold = 6000
mousecam_onsets = Widefield_General_Functions.get_step_onsets(mousecam_trace, threshold=threshold)
print("mousecma onsets", len(mousecam_onsets))

"""
scaled_mousecam_trace = np.divide(mousecam_trace, np.max(mousecam_trace))
plt.scatter(mousecam_onsets, np.ones(len(mousecam_onsets)))
plt.plot(scaled_mousecam_trace)
plt.show()
"""



# Load Videos
cam_1_file = r"/media/matthew/29D46574463D2856/Rig_2_mousecam_Test/led_test_3_2022-02-01-16-26-43_cam_1.mp4"
cam_2_file = r"/media/matthew/29D46574463D2856/Rig_2_mousecam_Test/led_test_3_2022-02-01-16-26-43_cam_2.mp4"

cam_1_frame_count, cam_1_even_frame_intensities, cam_1_odd_frame_intensities = load_video_as_numpy_array(cam_1_file)
cam_2_frame_count, cam_2_even_frame_intensities, cam_2_odd_frame_intensities = load_video_as_numpy_array(cam_2_file)

print("Cam 1 frames: ", cam_1_frame_count)
print("Cam 2 frames: ", cam_2_frame_count)

plt.title("Cam 1 frame intensities")
plt.plot(cam_1_even_frame_intensities)
plt.plot(cam_1_odd_frame_intensities)
plt.show()



plt.title("Cam 2 frame intensities")
plt.plot(cam_2_even_frame_intensities)
plt.plot(cam_2_odd_frame_intensities)
plt.show()

