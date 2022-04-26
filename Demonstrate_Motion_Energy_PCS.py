import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import os

def load_video_as_numpy_array(video_file, number_of_frames):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Frames: ", frameCount)


    # Extract Selected Frames
    frame_index = 0
    ret = True

    video_data = []
    while (frame_index < number_of_frames and ret):
        ret, frame = cap.read()
        video_data.append(frame)
        frame_index += 1
    cap.release()

    video_data = np.array(video_data)
    return video_data


number_of_frames = 1000
video_file = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging/NXAK4.1B_2021-04-08-10-49-26_cam_1.mp4"
video_data = load_video_as_numpy_array(video_file, number_of_frames)
video_data = video_data[:, :, :, 0]
print(np.shape(video_data))



video_max = np.max(video_data)

motion_energy = np.diff(video_data, axis=0)
motion_energy = np.abs(motion_energy)
max_motion_energy = np.max(motion_energy)

image_height = np.shape(video_data)[1]
image_width = np.shape(video_data)[2]

motion_energy = np.reshape(motion_energy, (number_of_frames-1, image_height * image_width))
model = PCA(n_components=20)
trajectories = model.fit_transform(motion_energy)

figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(3,1,1)
axis_2 = figure_1.add_subplot(3,1,2)
axis_3 = figure_1.add_subplot(3,1,3)
axis_1.plot(trajectories[2:, 0])
axis_2.plot(trajectories[2:, 1], c='r')
axis_3.plot(trajectories[2:, 2], c='g')
plt.show()


model_components = model.components_

reconstrcuted_data = np.dot(trajectories, model_components)
reconstrcuted_data = np.reshape(reconstrcuted_data, (number_of_frames-1, image_height, image_width))
print("reconstrcuuuuuuuuuuuuuuuuted data", np.shape(reconstrcuted_data))



figure_1 = plt.figure()
#plt.ion()
for frame_index in range(1000):
    video_axis = figure_1.add_subplot(1, 2, 1)
    energy_axis = figure_1.add_subplot(1, 2, 2)

    frame = video_data[frame_index]
    energy = reconstrcuted_data[frame_index]

    video_axis.imshow(frame, vmin=0, vmax=video_max, cmap='Greys_r')
    energy_axis.imshow(energy, vmin=0, vmax=max_motion_energy, cmap='Reds')

    video_axis.axis('off')
    energy_axis.axis('off')

    save_directory = os.path.join("/home/matthew/Documents/Thesis_Comitte_24_02_2022/Motion_PCS", str(frame_index).zfill(4) + ".png")
    plt.savefig(save_directory)

    #plt.show()
    plt.clf()
    #plt.draw()
    #plt.pause(0.1)
    #plt.clf()
