import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from PIL import Image
from time import clock
import os
import cv2
print("cv2 version", cv2.__version__)
from datetime import datetime



def check_led_colours(blue_array, violet_array):
    figure_1 = plt.figure()
    axes_1 = figure_1.subplots(1, 2)

    blue_image = blue_array[:, 0]
    blue_image = np.reshape(blue_image, (600,608))
    axes_1[0].set_title("Blue?")
    axes_1[0].imshow(blue_image)

    violet_image = violet_array[:, 0]
    violet_image = np.reshape(violet_image, (600,608))
    axes_1[1].set_title("Violet?")
    axes_1[1].imshow(violet_image)
    plt.show()


def reconstruct_sample_video(blue_array, violet_array, reconstructed_video_file):
    print("Reconstructing Video ")

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

    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    number_of_frames = np.shape(blue_array)[0]


    for frame in range(number_of_frames):

        blue_frame = blue_array[frame]
        blue_frame = np.ndarray.astype(blue_frame, np.uint8)

        violet_frame = violet_array[frame]
        violet_frame = np.ndarray.astype(violet_frame, np.uint8)

        blue_frame   = np.reshape(blue_frame, (600,608))
        violet_frame = np.reshape(violet_frame, (600, 608))

        #plt.imshow(blue_frame, cmap='jet', vmin=0, vmax=255)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()

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


def get_max_projection(array, home_directory):
    print("Getting Max Projection")

    sample = array[:, 0:1000]
    max_projection = np.max(sample, axis=1)
    max_projection = np.reshape(max_projection, (600, 608))

    plt.imshow(max_projection)
    plt.show()

    np.save(home_directory + "/max_projection", max_projection)

"""
def get_mask(fraction, home_directory):
    print("Getting Mask")

    image = np.load(home_directory + "/max_projection.npy")
    image_shape = [np.shape(image)[0], np.shape(image[1])]

    rows = image_shape[0]
    columns = np.shape(image)[1]
    max_value = np.max(image)
    min_value = np.min(image)
    mask = np.zeros((rows, columns))
    threshold = ((max_value - min_value) * fraction) + min_value

    for y in range(rows):
        for x in range(columns):
            if image[y][x] > threshold:
                mask[y][x] = 1

    print("Active pixels: ", np.sum(mask))
    # plt.imshow(mask)
    # plt.show()
    np.save(home_directory + "/Mask", mask)
"""

def get_filter_coefficients():
    # Create Butterwoth Bandpass Filter
    sampling_frequency = 25  # In Hertz
    cutoff_frequency = 8.5  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    low_cutoff_frequency = 0.01
    w_low = low_cutoff_frequency / (sampling_frequency / 2)
    b, a = signal.butter(2, [w_low, w], 'bandpass')

    return b, a


def process_pixels(blue_data, violet_data, output_file, home_directory, bandpass_filter=True, heamocorrection_type="Regression"):
    print("Processing Pixels")

    # Load Mask:
    mask = np.load(home_directory + "/mask.npy")

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)


    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)
    indicies = list(indicies)

    # Get Butterworth Filter Coefficients
    b, a, = get_filter_coefficients()

    # Load Data
    blue_matrix = h5py.File(blue_data, 'r')
    violet_matrix = h5py.File(violet_data, 'r')

    number_of_pixels = np.shape(blue_matrix["Data"])[0]
    number_of_images = np.shape(blue_matrix["Data"])[1]
    number_of_active_pixels = np.sum(flat_mask)


    # Create Lists to Store Max and Min Values
    min_values = []
    max_values = []

    # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_active_pixels)

    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_active_pixels, number_of_images), dtype=np.float32, chunks=True,
                                   compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks)
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])

            chunk_indicies = indicies[chunk_start:chunk_stop]

            blue_data = blue_matrix["Data"][chunk_indicies]
            violet_data = violet_matrix["Data"][chunk_indicies]

            if bandpass_filter == True:
                blue_data, violet_data = perform_bandpass_filter(blue_data, violet_data, b, a)

            if heamocorrection_type == "Regression":
                processed_data = heamocorrection_regression(blue_data, violet_data)

            elif heamocorrection_type == "Ratio":
                processed_data = heamocorrection_ratio(blue_data, violet_data)

            else:
                print("Invalid Heamocorrection Type!")

            chunk_min_vector = np.percentile(processed_data, axis=1, q=5)
            chunk_max_vector = np.percentile(processed_data, axis=1, q=95)

            for value in chunk_min_vector:
                min_values.append(value)
            for value in chunk_max_vector:
                max_values.append(value)

            print("Procesed data shape", np.shape(processed_data))
            dataset[chunk_start:chunk_stop, :] = processed_data

    min_vector = np.array(min_values)
    max_vector = np.array(max_values)

    np.save(home_directory + "/Pixel_Baseline_Values.npy",   min_vector)
    np.save(home_directory + "/Pixel_Max_Values.npy",        max_vector)

    print("Finished processing pixels")

def perform_bandpass_filter(blue_data, violet_data, b, a):
    blue_data = signal.filtfilt(b, a, blue_data, axis=1)
    violet_data = signal.filtfilt(b, a, violet_data, axis=1)

    return blue_data, violet_data


def heamocorrection_regression(blue_data, violet_data):
    # Perform Regression
    chunk_size = np.shape(blue_data)[0]
    for pixel in range(chunk_size):

        violet_trace = violet_data[pixel]
        blue_trace = blue_data[pixel]

        slope, intercept, r, p, stdev = stats.linregress(violet_trace, blue_trace)
        violet_trace = np.multiply(violet_trace, slope)
        blue_trace = np.subtract(blue_trace, violet_trace)

        blue_min = np.min(blue_trace)
        if blue_min < 0:
            blue_trace = np.add(blue_trace, abs(blue_min))

        blue_data[pixel] = blue_trace

    return blue_data


def heamocorrection_ratio(blue_data, violet_data):
    print("Blue data", np.shape(blue_data))

    ratio = np.divide(blue_data, violet_data)
    mean_ratio = np.mean(ratio, axis=1)

    print("Mean ratio", np.shape(mean_ratio))
    corrected_data = ratio / mean_ratio[:, None]
    return corrected_data


def perform_delta_f_on_dataset(base_directory, output_file):

    print("Getting Delta F: ", base_directory)

    # Load Min and Max Vectors
    min_vector_file = base_directory + "/Pixel_Baseline_Values.npy"
    max_vector_file = base_directory + "/Pixel_Max_Values.npy"
    min_vector = np.load(min_vector_file, allow_pickle=True)
    max_vector = np.load(max_vector_file, allow_pickle=True)

    # Load Preprocessed Data
    processed_data_file_location = base_directory + "/Preprocessed_Data_Pixelwise.hdf5"
    processed_data_file = h5py.File(processed_data_file_location, 'r')
    preprocessed_data = processed_data_file["Data"]

    print("Prepriocessed Data Shape", np.shape(preprocessed_data))
    number_of_images = np.shape(preprocessed_data)[1]
    number_of_active_pixels = np.shape(preprocessed_data)[0]

     # Define Chunking Settings
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_active_pixels)

    # Process Data
    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_active_pixels, number_of_images), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks)
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])

            chunk_min_vector = min_vector[chunk_start:chunk_stop]
            chunk_max_vector = max_vector[chunk_start:chunk_stop]

            # Load Preprocessed Data
            preprocessed_chunk = preprocessed_data[chunk_start:chunk_stop, :]

            # Perform Delta F
            processed_chunk = calculate_delta_f(preprocessed_chunk, chunk_min_vector)

            # Normalise Delta F
            processed_chunk = normalise_delta_f(processed_chunk, chunk_max_vector, chunk_min_vector)

             # Save Data
            dataset[chunk_start:chunk_stop, :] = processed_chunk



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


def normalise_delta_f(delta_f_matrix, max_pixel_values, min_pixel_values):

    # Transpose Baseline Vector so it can be used by numpy subtract
    max_pixel_values = max_pixel_values[:, np.newaxis]
    min_pixel_values = min_pixel_values[:, np.newaxis]

    # First Calculate Maximum Possible Delta F
    max_delta_f = np.subtract(max_pixel_values, min_pixel_values)
    max_delta_f_over_f = np.divide(max_delta_f, min_pixel_values)

    # Divide Pixels by Maximum Delta F
    delta_f_matrix = np.divide(delta_f_matrix, max_delta_f_over_f)

    return delta_f_matrix


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
    sample_data = processed_data[:, 1000:1000 + sample_size]
    sample_data = np.nan_to_num(sample_data)
    sample_data = np.transpose(sample_data)

    # Get Colour Boundaries
    cm = plt.cm.ScalarMappable(norm=None, cmap='inferno')

    colour_max = 1
    colour_min = 0

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

        colored_image = cm.to_rgba(image)
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def reshape_processed_data(processed_file_location, framewise_file):
    print("Reshaping Processed Data")

    # Load Processed Data File
    processed_data_file = h5py.File(processed_file_location, 'r')
    processed_data = processed_data_file["Data"]

    # Get Data Dimensions
    number_of_frames = np.shape(processed_data)[1]
    print("number of frames", number_of_frames)

    number_of_pixels = np.shape(processed_data)[0]
    print("Number of pixels", number_of_pixels)

    # Create Pytables File To Save Data
    framewise_file_object = tables.open_file(framewise_file, mode='w')
    framewise_file_earray = framewise_file_object.create_earray(framewise_file_object.root,
                                                                name='Data',
                                                                atom=tables.Float32Atom(),
                                                                shape=(0, number_of_pixels),
                                                                expectedrows=number_of_frames)

    # Get Chunking Structure
    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size,
                                                                                   number_of_frames)

    # Restructure Data
    for chunk_index in range(number_of_chunks):
        print("Chunk:", str(chunk_index).zfill(2), " of ", number_of_chunks)
        chunk_size = chunk_sizes[chunk_index]
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        data = processed_data[:, chunk_start:chunk_stop]
        data = np.transpose(data)

        for frame in range(chunk_size):
            framewise_file_earray.append([data[frame]])

        framewise_file_object.flush()

    processed_data_file.close()
    framewise_file_object.close()



def get_blue_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Blue" in file:
            return base_directory + "/" + file

def get_violet_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Violet" in file:
            return base_directory + "/" + file


def perform_heamocorrection(home_directory):
    # Preprocessing Variables
    bandpass_filter = True
    heamocorrection_type = "Regression"

    # Assign File Locations
    blue_file                   = get_blue_file(home_directory)
    violet_file                 = get_violet_file(home_directory)
    reconstructed_video_file    = home_directory + "/Greyscale_Reconstruction.avi"
    preprocessed_data_pixelwise = home_directory + "/Preprocessed_Data_Pixelwise.hdf5"
    delta_f_pixelwise           = home_directory + "/delta_f_pixelwise.hdf5"
    delta_f                     = home_directory + "/Delta_F.h5"

    #Load Data
    blue_data_container = h5py.File(blue_file, 'r')
    blue_data           = blue_data_container["Data"]

    violet_data_container = h5py.File(violet_file, 'r')
    violet_data           = violet_data_container["Data"]

    # Reconstruct Greyscale Video
    print("reconstructed video file", reconstructed_video_file)
    reconstruct_sample_video(blue_data, violet_data, reconstructed_video_file)

    # Extract Signal
    process_pixels(blue_file, violet_file, preprocessed_data_pixelwise, home_directory, bandpass_filter=bandpass_filter,heamocorrection_type=heamocorrection_type)

    # Perform Delta F
    perform_delta_f_on_dataset(home_directory, delta_f_pixelwise)

    # Create Sample Video
    create_sample_video(delta_f_pixelwise, home_directory, blur_size=2)

    # Reshape Data
    reshape_processed_data(delta_f_pixelwise, delta_f)




#perform_heamocorrection("/home/matthew/Documents/NRXN_Data/NRXN78.1A/2020_11_21_Discrimination")
#perform_heamocorrection("/home/matthew/Documents/NRXN_Data/NRXN51.1C/2020_11_27_Switching")
#perform_heamocorrection(r"/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Analysis/78.1D_2020_11_30_Switching")
#perform_heamocorrection(r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/2020_11_17/NRXN79.1A/1")