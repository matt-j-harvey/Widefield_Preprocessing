import numpy as np
import matplotlib.pyplot as plt
import h5py
import tables
from scipy import signal, ndimage, stats
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time



import Preprocessing_Utils
import Get_Baseline_Frames

def get_session_name(base_directory):

    # Split File Path By Forward Slash
    split_base_directory = base_directory.split("/")

    # Take The Last Two and Join By Underscore
    session_name = split_base_directory[-2] + "_" + split_base_directory[-1]

    return session_name


def get_motion_corrected_data_filename(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Motion_Corrected_Mask_Data" in file:
            return file


def load_mask(base_directory):

    # Load Mask
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))
    image_height, image_width = np.shape(mask)
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)

    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def reconstruct_sample_video(base_directory, save_directory):
    print("Reconstructing Sample Video For Session", base_directory)

    # Load Data
    motion_corrected_data_file = get_motion_corrected_data_filename(base_directory)
    data_file = os.path.join(base_directory, motion_corrected_data_file)
    data_container = h5py.File(data_file, 'r')
    blue_array = data_container["Blue_Data"]
    violet_array = data_container["Violet_Data"]

    # Take Sample of Data
    blue_array   = blue_array[:, 1000:2000]
    violet_array = violet_array[:, 1000:2000]

    # Transpose Data
    blue_array = np.transpose(blue_array)
    violet_array = np.transpose(violet_array)

    # Convert From 16 bit to 8 bit
    blue_array   = np.divide(blue_array, 65536)
    violet_array = np.divide(violet_array, 65536)

    blue_array = np.multiply(blue_array, 255)
    violet_array = np.multiply(violet_array, 255)

    # Get Original Pixel Dimensions
    frame_width = 608
    frame_height = 600

    # Load Mask
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))
    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)

    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    # Create Video File
    reconstructed_video_file = os.path.join(save_directory, "Greyscale_Reconstruction.avi")
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

def get_lowcut_coefs(w=0.0033, fs=28.):
    b, a = signal.butter(2, w/(fs/2.), btype='highpass');
    return b, a


def perform_lowcut_filter(data, b, a):
    filtered_data = signal.filtfilt(b, a, data, padlen=10000)
    return filtered_data


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



def calculate_delta_f(activity_matrix, exclusion_point):

    # Get Baseline
    baseline_vector = np.mean(activity_matrix[:, exclusion_point:], axis=1)

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
    sample_size = 2000
    sample_data = processed_data[3000:3000 + sample_size]
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

    return data_sample, baseline, max_vector


def remove_early_frames(frame_onsets, exclusion_point):

    thresholded_onsets = []

    for onset in frame_onsets:
        if onset > exclusion_point:
            thresholded_onsets.append(onset)

    return thresholded_onsets


def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros(image_height * image_width)
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image


def convert_to_int(chunk_data):

    # Remove Nans
    chunk_data = np.nan_to_num(chunk_data)

    # Multiply By Uint36 Limit
    chunk_data = np.multiply(chunk_data, 65535)

    # Clip
    chunk_data = np.clip(chunk_data, a_min=0, a_max=65535)

    return chunk_data


def smooth_data(chunk_data,  indicies, image_height, image_width, filter_width):

    # Remove Nans
    chunk_data = np.nan_to_num(chunk_data)

    smoothed_data = []
    for frame in chunk_data:
        reconstructed_frame = create_image_from_data(frame, indicies, image_height, image_width)
        reconstructed_frame = gaussian_filter(reconstructed_frame, sigma=filter_width)
        reconstructed_frame = np.reshape(reconstructed_frame, (image_height * image_width))
        frame = reconstructed_frame[indicies]
        smoothed_data.append(frame)

    smoothed_data = np.array(smoothed_data)
    return smoothed_data



def create_sample_video_integer(base_directory, save_directory):

    print("Creating Sample Delta F Video")

    # Load Mask
    indicies, frame_height, frame_width = load_mask(base_directory)

    # Load Processed Data
    delta_f_file_location = os.path.join(save_directory, "Delta_F.h5")
    delta_f_file = tables.open_file(delta_f_file_location, mode='r')
    processed_data = delta_f_file.root.Data

    # Get Sample Data
    start_time = 10000
    sample_size = 3000
    sample_data = processed_data[start_time:start_time + sample_size]
    sample_data = np.nan_to_num(sample_data)

    # Filter
    sampling_frequency = 28  # In Hertz
    cutoff_frequency = 12  # In Hertz
    w = cutoff_frequency / (sampling_frequency / 2)  # Normalised frequency
    b, a = signal.butter(1, w, 'lowpass')
    sample_data = signal.filtfilt(b, a, sample_data, axis=0)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(sample_data)
    sample_data = model.inverse_transform(transformed_data)

    # Get Colour Map
    colourmap = Preprocessing_Utils.get_musall_cmap()
    cm = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
    colour_magnitude = 0.05
    cm.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

    # Get Background Pixels
    #background_pixels = get_background_pixels(indicies, frame_height, frame_width)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    # Create Video File
    video_name = os.path.join(save_directory, "Movie_Baseline.avi")
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
        image = ndimage.gaussian_filter(image, 1)

        # Set Image Colours
        colored_image = cm.to_rgba(image)
        #colored_image[background_pixels] = [1, 1, 1, 1]
        colored_image = colored_image * 255

        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    delta_f_file.close()


class metadata_particle(tables.IsDescription):

    session_name = tables.StringCol(100)
    ai_filename = tables.StringCol(100)
    violet_baseline_frames = tables.UInt64Col()
    blue_baseline_frames = tables.UInt64Col()
    lowcut_filter = tables.BoolCol()
    lowcut_freq = tables.FloatCol()
    exclusion_point = tables.UInt64Col()
    gaussian_filter = tables.BoolCol()
    gaussian_filter_width = tables.UInt64Col()
    pixel_baseline_list = tables.FloatCol()
    pixel_maximum_list = tables.FloatCol()



def save_session_metadata(base_directory, delta_f_file, violet_baseline_frames, blue_baseline_frames, lowcut_filter, lowcut_freq, exclusion_point, gaussian_filter_width, pixel_baseline_list, pixel_maximum_list):

    # Add Metadata
    session_name = get_session_name(base_directory)
    ai_filename = Preprocessing_Utils.get_ai_filename(base_directory)
    metadata_table = delta_f_file.create_table(where=delta_f_file.root, name='metadata_table', description=metadata_particle, title="metadata_table")
    metadata_row = metadata_table.row
    metadata_row['session_name'] = session_name
    metadata_row['ai_filename'] = ai_filename
    metadata_row['lowcut_filter'] = lowcut_filter
    metadata_row['lowcut_freq'] = lowcut_freq
    metadata_row['exclusion_point'] = exclusion_point
    metadata_row['gaussian_filter'] = gaussian_filter
    metadata_row['gaussian_filter_width'] = gaussian_filter_width
    metadata_row.append()
    metadata_table.flush()

    # Add Baseline Frames
    if violet_baseline_frames!= None:
        delta_f_file.create_array(delta_f_file.root, 'violet_baseline_frames', np.array(violet_baseline_frames), "violet_baseline_frames")
        delta_f_file.create_array(delta_f_file.root, 'blue_baseline_frames', np.array(blue_baseline_frames), "blue_baseline_frames")

    # Add Pixel Baselines and Pixel Maximums
    delta_f_file.create_array(delta_f_file.root, 'pixel_baseline_list', np.array(pixel_baseline_list), "pixel_baseline_list")
    delta_f_file.create_array(delta_f_file.root, 'pixel_maximum_list', np.array(pixel_maximum_list), "pixel_maximum_list")



def get_baseline_mean_and_sd(processed_data, baseline_frames):
    processed_data = np.transpose(processed_data)
    baseline_mean = np.nanmean(processed_data[baseline_frames], axis=0)
    baseline_sd = np.nanstd(processed_data[baseline_frames], axis=0)
    return baseline_mean, baseline_sd


def process_chunk(chunk_start, chunk_stop, blue_matrix, violet_matrix, lowcut_filter, b, a, exclusion_point, blue_baseline_frames):

    # Load Chunk Data
    #start_time = time.time()
    blue_data = blue_matrix[chunk_start:chunk_stop]
    violet_data = violet_matrix[chunk_start:chunk_stop]
    #end_time = time.time()
    #print("Loading Data Time", end_time - start_time)

    # Remove NaNs
    #start_time = time.time()
    blue_data = np.nan_to_num(blue_data)
    violet_data = np.nan_to_num(violet_data)
    #end_time = time.time()
    #print("remove nan Time", end_time - start_time)

    # Calculate Delta F
    #start_time = time.time()
    blue_data = calculate_delta_f(blue_data, exclusion_point)
    violet_data = calculate_delta_f(violet_data, exclusion_point)
    #end_time = time.time()
    #print("Calculate Delta F", end_time - start_time)

    # Lowcut Filter
    #start_time = time.time()
    if lowcut_filter == True:
        blue_data[:, exclusion_point:] = perform_lowcut_filter(blue_data[:, exclusion_point:], b, a)
        violet_data[:, exclusion_point:] = perform_lowcut_filter(violet_data[:, exclusion_point:], b, a)
    #end_time = time.time()
    #print("Lowcut Filter", end_time - start_time)

    # Regression and Subtraction
    #start_time = time.time()
    processed_data = heamocorrection_regression(blue_data, violet_data)
    #end_time = time.time()
    #print("Regression + Subtraction", end_time - start_time)

    # Get Mean and SD - For Potential Later Z Scoring
    #start_time = time.time()
    baseline_mean, baseline_sd = get_baseline_mean_and_sd(processed_data, blue_baseline_frames)
    #end_time = time.time()
    #print("Getting Baseline Mean and SD", end_time - start_time)

    # Transpose
    #start_time = time.time()
    processed_data = np.transpose(processed_data)
    #end_time = time.time()
    #print("Transposing", end_time - start_time)

    # Convert to 32 Bit Float
    #start_time = time.time()
    processed_data = np.ndarray.astype(processed_data, np.float32)
    #end_time = time.time()
    #print("FLoat 32 Conversion Time", end_time - start_time)

    # Convert Baseline and SD to Lists
    baseline_mean = list(baseline_mean)
    baseline_sd = list(baseline_sd)

    return processed_data, baseline_mean, baseline_sd



def process_pixels(base_directory, output_directory, exclusion_point=3000, lowcut_filter=True, low_cut_freq=0.0033, gaussian_filter=True, gaussian_filter_width=1):

    """
    Order of operations taken from Anne Churchland Group Github:  https://github.com/churchlandlab/wfield/tree/master/wfield
    Also See Paper: Chronic, cortex-wide imaging of specific cell populations during behavior - Joao Couto - Nat Protoc. 2021 Jul; 16(7): 3241–3263. - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8788140/
    Data analysis (Stage 4) pg 9

    Steps
    Data should already be motion corrected
    1 - Delta F Over F (F is mean value over trial)
    2 - Lowpass Filtering
    3 - Regression And Subtraction

    I also calculate the Mean and SD Of Each Pixel In the Baseline Periods Prior To Stimuli Onsets and Save These, This Allows Me to Z Score The Data Later If I Want

    This First 2-3 Mins (approx 3000 frames) The LEDs Are Initially Quite Bright And Then Dim, I Think This Is Due to Heating Effects, So I Exclude THe First 2-3 Mins of Each Session
    """

    # Get Butterworth Filter Coefficients
    b, a, = get_lowcut_coefs(w=low_cut_freq)

    # Get Filenames
    intermediate_delta_f = os.path.join(output_directory, "Delta_F.hdf5")
    final_delta_f = os.path.join(output_directory, "Delta_F.h5")

    # Load Data
    motion_corrected_filename = get_motion_corrected_data_filename(base_directory)
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]

    # Load Baseline Frame Indexes
    violet_baseline_frames = np.load(os.path.join(output_directory, "Violet_Baseline_Frames.npy"))
    blue_baseline_frames = np.load(os.path.join(output_directory, "Blue_Baseline_Frames.npy"))

    # Remove Onsets Prior To Exclusion Points
    violet_baseline_frames = remove_early_frames(violet_baseline_frames, exclusion_point)
    blue_baseline_frames = remove_early_frames(blue_baseline_frames, exclusion_point)

    # Get Data Structure
    number_of_pixels, number_of_images = np.shape(blue_matrix)
    print("Number of images", number_of_images)
    print("number of pixels", number_of_pixels)

    # Define Chunking Settings
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_pixels)

    # Create List To Store Pixel Baseline SDs so We Can Z Score Later If We Wish
    pixel_baseline_sd_list = []
    pixel_baseline_mean_list = []

    print("Heamocorrecting")
    with h5py.File(intermediate_delta_f, "w") as f:
       dataset = f.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)

       for chunk_index in tqdm(range(number_of_chunks)):
           chunk_start = int(chunk_starts[chunk_index])
           chunk_stop = int(chunk_stops[chunk_index])

           # Process This Chunk
           processed_data, baseline_mean, baseline_sd = process_chunk(chunk_start, chunk_stop, blue_matrix, violet_matrix, lowcut_filter, b, a, exclusion_point, blue_baseline_frames)

           # Add Values To Baseline Mean
           pixel_baseline_mean_list = pixel_baseline_mean_list + baseline_mean
           pixel_baseline_sd_list = pixel_baseline_sd_list + baseline_sd

           # Insert Back
           #start_time = time.time()
           dataset[exclusion_point:, chunk_start:chunk_stop] = processed_data[exclusion_point:]
           #end_time = time.time()
           #print("Writing Data Time", end_time - start_time)

    # Close Motion Corrected Data
    motion_corrected_data_container.close()

    # Define Chunking Settings
    preferred_chunk_size = 10000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, number_of_images)

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)

    # Open Intermediate Delta F File
    intermediate_delta_f_container = h5py.File(intermediate_delta_f, 'r')
    intermediate_delta_f_matrix = intermediate_delta_f_container["Data"]

    # Create Tables File
    delta_f_file = tables.open_file(final_delta_f, mode='w')

    # Save Session Metadata
    pixel_baseline_sd_list = np.array(pixel_baseline_sd_list)
    pixel_baseline_mean_list = np.array(pixel_baseline_mean_list)
    save_session_metadata(base_directory, delta_f_file, violet_baseline_frames, blue_baseline_frames, lowcut_filter, low_cut_freq, exclusion_point, gaussian_filter_width, pixel_baseline_sd_list, pixel_baseline_mean_list)

    # Add Delta F Data
    delta_f_storage = delta_f_file.create_earray(delta_f_file.root, "Data", tables.Float32Atom(), shape=(0, number_of_pixels), expectedrows=number_of_images)

    print("Reshaping")
    for chunk_index in tqdm(range(number_of_chunks)):

       # Load Chunk Data
       chunk_start = int(chunk_starts[chunk_index])
       chunk_stop = int(chunk_stops[chunk_index])
       chunk_data = intermediate_delta_f_matrix[chunk_start:chunk_stop]

       # Smooth Data
       if gaussian_filter == True:
           chunk_data = smooth_data(chunk_data, indicies, image_height, image_width, gaussian_filter_width)

       for frame in chunk_data:
           delta_f_storage.append([frame])

       delta_f_storage.flush()

    # Close Open Files
    delta_f_file.close()
    intermediate_delta_f_container.close()

    # Delete Intermediate File
    os.remove(intermediate_delta_f)



def perform_heamocorrection(base_directory, save_directory, exclusion_point=3000, lowcut_filter=True, low_cut_freq=0.0033, gaussian_filter=True, gaussian_filter_width=1, use_baseline_frames=True):

    # Reconstruct Greyscale Video
    print("Base Directory", base_directory)
    print("Save Directory", save_directory)
    reconstruct_sample_video(base_directory, save_directory)

    # Get Baseline Frames
    Get_Baseline_Frames.get_baseline_frames(base_directory, save_directory)

    # Extract Signal
    process_pixels(base_directory, save_directory, exclusion_point=exclusion_point, lowcut_filter=lowcut_filter, low_cut_freq=low_cut_freq, gaussian_filter=gaussian_filter, gaussian_filter_width=gaussian_filter_width)

    # Create Sample Video
    create_sample_video_integer(base_directory, save_directory)

