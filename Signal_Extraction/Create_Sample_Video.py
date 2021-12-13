import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


def create_sample_video(processed_file_location, home_directory, blur_size=1):
    print("Creating Sample Video")

    # Load Mask
    mask = np.load(home_directory + "/mask.npy")
    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    # Load Processed Data
    processed_data_file = h5py.File(processed_file_location, 'r')
    processed_data = processed_data_file["Data"]

    # Get Sample Data
    sample_size = 7000
    sample_data = processed_data[1000:1000 + sample_size]
    sample_data = np.nan_to_num(sample_data)


    # Get Colour Boundaries
    cm = plt.cm.ScalarMappable(norm=None, cmap='inferno')

    colour_max = 1
    colour_min = 0

    cm.set_clim(vmin=colour_min, vmax=colour_max)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    video_name = home_directory + "/Movie_Baseline_Combined.avi"
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width, frame_height), fps=30)  # 0, 12


    window_size = 2
    for frame in range(sample_size - window_size):  # number_of_files:
        template = np.zeros((frame_height * frame_width))

        image = sample_data[frame:frame + window_size]
        image = np.mean(image, axis=0)
        image = np.nan_to_num(image)
        np.put(template, indicies, image)
        image = np.reshape(template, (frame_height, frame_width))
        image = ndimage.gaussian_filter(image, blur_size)

        colored_image = cm.to_rgba(image)
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()