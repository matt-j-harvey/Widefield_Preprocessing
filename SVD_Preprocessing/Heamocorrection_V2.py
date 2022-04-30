import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal, ndimage, stats
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


def get_session_name(base_directory):

    # Split File Path By Forward Slash
    split_base_directory = base_directory.split("/")

    # Take The Last Two and Join By Underscore
    session_name = split_base_directory[-2] + "_" + split_base_directory[-1]

    return session_name



def reconstruct_sample_video(base_directory, reconstructed_video_file):
    print("Reconstructing Raw Video Sample")

    # Load Data
    data_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    data_container = h5py.File(data_file, 'r')
    blue_array = data_container["Blue_Data"]
    violet_array = data_container["Violet_Data"]

    #Take Sample of Data
    blue_array   = blue_array[:, 1000:2000]
    violet_array = violet_array[:, 1000:2000]

    #Transpose Data
    blue_array = np.transpose(blue_array)
    violet_array = np.transpose(violet_array)

    # Convert From 16 bit to 8 bit
    blue_array   = np.divide(blue_array, 65536)
    violet_array = np.divide(violet_array, 65536)

    blue_array = np.multiply(blue_array, 255)
    violet_array = np.multiply(violet_array, 255)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    # Load Mask
    mask = np.load(base_directory + "/Generous_Mask.npy")
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)

    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    number_of_frames = np.shape(blue_array)[0]


    for frame in range(number_of_frames):

        blue_template = np.zeros(frame_height * frame_width)
        violet_template = np.zeros(frame_height * frame_width)

        blue_frame = blue_array[frame]
        violet_frame = violet_array[frame]

        blue_template[indicies] = blue_frame
        violet_template[indicies] = violet_frame

        blue_template = np.ndarray.astype(blue_template, np.uint8)
        violet_template = np.ndarray.astype(violet_template, np.uint8)

        blue_frame   = np.reshape(blue_template, (600,608))
        violet_frame = np.reshape(violet_template, (600, 608))

        image = np.hstack((violet_frame, blue_frame))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops


def get_filter_coefficients():

    # Create Butterwoth Bandpass Filter
    sampling_frequency = 25  # In Hertz
    cutoff_frequency = 12  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    low_cutoff_frequency = 0.1
    w_low = low_cutoff_frequency / (sampling_frequency / 2)
    b, a = signal.butter(2, [w_low, w], 'bandpass')

    return b, a


def perform_bandpass_filter(blue_data, violet_data, b, a):
    blue_data = signal.filtfilt(b, a, blue_data, axis=1)
    violet_data = signal.filtfilt(b, a, violet_data, axis=1)
    return blue_data, violet_data


def heamocorrection_regression(blue_data, violet_data):

    # Perform Regression
    chunk_size = np.shape(blue_data)[0]
    for pixel in range(chunk_size):

        # Load Pixel Traces
        violet_trace = violet_data[pixel]
        blue_trace = blue_data[pixel]

        # Perform Regression
        slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)

        # Scale Violet Trace
        violet_trace = np.multiply(violet_trace, slope)
        violet_trace = np.add(violet_trace, intercept)

        # Subtract From Blue Trace
        blue_trace = np.subtract(blue_trace, violet_trace)

        # Insert Back Corrected Trace
        blue_data[pixel] = blue_trace

    return blue_data


def calculate_delta_f(activity_matrix, baseline_vector):

    # Transpose Baseline Vector so it can be used by numpy subtract
    baseline_vector = baseline_vector[:, np.newaxis]

    # Get Delta F
    delta_f = np.subtract(activity_matrix, baseline_vector)

    # Remove Negative Values
    delta_f = np.clip(delta_f, a_min=0, a_max=None)

    # Divide by baseline
    delta_f_over_f = np.divide(delta_f, baseline_vector)

    # Remove NANs
    delta_f_over_f = np.nan_to_num(delta_f_over_f)

    return delta_f_over_f



def create_sample_video(processed_file_location, home_directory, blur_size=1):
    print("Creating Sample Delta F Video")

    # Load Mask
    mask = np.load(home_directory + "/Generous_Mask.npy")
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

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(sample_data)
    sample_data = model.inverse_transform(transformed_data)

    # Get Colour Boundaries
    cm = plt.cm.ScalarMappable(norm=None, cmap='inferno')

    colour_max = 0.7
    colour_min = 0.1

    cm.set_clim(vmin=colour_min, vmax=colour_max)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    video_name = home_directory + "/Movie_Baseline.avi"
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width, frame_height), fps=30)  # 0, 12

    # plt.ion()
    window_size = 3

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


def normalise_traces(data_sample):

    # Subtract Baseline
    baseline = np.min(data_sample, axis=1)
    baseline = baseline[:, np.newaxis]
    data_sample = np.subtract(data_sample, baseline)

    # Divide By Max
    max_vector = np.max(data_sample, axis=1)
    max_vector = max_vector[:, np.newaxis]
    data_sample = np.divide(data_sample, max_vector)

    return data_sample


def view_sample(processed_file_location, home_directory, blur_size=1):
    print("Creating Sample Delta F Video")

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

    # Create Colourmap
    cmp = LinearSegmentedColormap.from_list('mycmap', [
        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),


        (0, 0, 0, 1),

        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    # Get Colour Boundaries
    cm = plt.cm.ScalarMappable(norm=None, cmap=cmp)

    colour_max = 0.1
    colour_min = -0.1

    cm.set_clim(vmin=colour_min, vmax=colour_max)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    video_name = home_directory + "/Movie_Baseline.avi"
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width, frame_height), fps=30)  # 0, 12

    # plt.ion()
    window_size = 2

    for frame in range(sample_size - window_size):  # number_of_files:
        template = np.zeros((frame_height * frame_width))

        image = sample_data[frame:frame + window_size]
        image = np.mean(image, axis=0)
        image = np.nan_to_num(image)
        np.put(template, indicies, image)
        image = np.reshape(template, (frame_height, frame_width))
        image = ndimage.gaussian_filter(image, blur_size)

        #plt.imshow(image, cmap=cmp)
        #plt.show()

        colored_image = cm.to_rgba(image)
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def process_pixels(base_directory, delta_f_file):

    print("Processing Pixels")
    # Mussal Order
    # 1 - Delta F Over F
    # 2 - Bandpass Filtering
    # 3 - Regression And Subtraction

    # Get Butterworth Filter Coefficients
    b, a, = get_filter_coefficients()

    # Load Data
    data_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    data_container = h5py.File(data_file, 'r')
    blue_matrix = data_container["Blue_Data"]
    violet_matrix = data_container["Violet_Data"]

    # Load Baseline Frame Indexes
    violet_baseline_frames = np.load(os.path.join(base_directory, "Violet_Baseline_Frames.npy"))
    blue_baseline_frames = np.load(os.path.join(base_directory, "Blue_baseline_frames.npy"))

    # Get Data Structure
    number_of_images = np.shape(blue_matrix)[1]
    number_of_pixels = np.shape(blue_matrix)[0]

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_pixels)

    with h5py.File(delta_f_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks, "at", datetime.now())

            # Load Chunk Data
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])
            blue_data = blue_matrix[chunk_start:chunk_stop]
            violet_data = violet_matrix[chunk_start:chunk_stop]
            print("loaded data", datetime.now())

            # Perform Delta F
            blue_baseline_data = blue_data[:, blue_baseline_frames]
            violet_baseline_data = violet_data[:, violet_baseline_frames]

            blue_baseline = np.percentile(blue_baseline_data, axis=1, q=5)
            violet_baseline = np.percentile(violet_baseline_data, axis=1, q=5)

            #blue_baseline = np.mean(blue_data, axis=1)
            #violet_baseline = np.mean(violet_data, axis=1)
            blue_data = calculate_delta_f(blue_data, blue_baseline)
            violet_data = calculate_delta_f(violet_data, violet_baseline)

            # Bandpass Filter
            blue_data, violet_data = perform_bandpass_filter(blue_data, violet_data, b, a)

            # Regression and Subtraction
            processed_data = heamocorrection_regression(blue_data, violet_data)

            # Normalise Delta F
            processed_data = normalise_traces(processed_data)

            # Insert Back
            dataset[:, chunk_start:chunk_stop] = np.transpose(processed_data)



def perform_heamocorrection(base_directory):

    # Assign File Locations
    reconstructed_video_file    = base_directory + "/Greyscale_Reconstruction.avi"
    delta_f_file                = base_directory + "/Delta_F.hdf5"

    # Reconstruct Greyscale Video
    reconstruct_sample_video(base_directory, reconstructed_video_file)

    # Extract Signal
    process_pixels(base_directory, delta_f_file)

    # Create Sample Video
    create_sample_video(delta_f_file, base_directory, blur_size=2)
