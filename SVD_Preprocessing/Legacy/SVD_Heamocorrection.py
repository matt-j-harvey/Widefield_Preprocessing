import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import TruncatedSVD
import h5py
import datetime
from tqdm import tqdm
import sys
from scipy import ndimage

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions


def lowpass(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen = 50)

def highpass(X, w = 3., fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen = 50)

def load_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


def view_blue_components(base_directory):

    # Load Blue Componnets
    blue_components = np.load(os.path.join(base_directory, "Blue_Data_SVD_incremental_components_200.npy"))

    # Load Mask
    indicies, image_height, image_width = load_mask(base_directory)


    number_of_components = 20

    figure_1 = plt.figure()
    rows = 4
    columns = 5
    for x in range(number_of_components):
        axis = figure_1.add_subplot(rows, columns, x+1)

        component_data = blue_components[x]
        component_image = Widefield_General_Functions.create_image_from_data(component_data, indicies, image_height, image_width)
        axis.imshow(component_image)

    plt.show()


def decompose_violet_video(base_directory):

    # Load Blue Componnets
    blue_components = np.load(os.path.join(base_directory, "Blue_Data_SVD_incremental_components_200.npy"))

    # Invert Blue Components
    inverse_components = np.linalg.pinv(blue_components)

    # Load Violet Data
    violet_file_object = h5py.File(os.path.join(base_directory, "Masked_Violet_Data.hdf5"), 'r')
    data_matrix = violet_file_object["Data"]
    number_of_frames, number_of_pixels = np.shape(data_matrix)

    # Get Chunk Structure
    chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(chunk_size, number_of_frames)

    violet_transformed_data = []
    for chunk_index in range(number_of_chunks):
        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]

        chunk_transformed_data = np.dot(np.transpose(inverse_components), np.transpose(chunk_data))
        print("Chunk Transformed Data Shape", np.shape(chunk_transformed_data))

        violet_transformed_data.append(chunk_transformed_data)

    violet_transformed_data = np.hstack(violet_transformed_data)

    # Save These
    np.save(os.path.join(output_directory, "Violet_Transformed_Data.npy"), violet_transformed_data)


def view_violet_reconstruction(output_directory):

    # Load Blue Components
    blue_components = np.load(os.path.join(output_directory, "Blue_Data_SVD_incremental_components_200.npy"))

    # Load Violet Loadings
    violet_loadings = np.load(os.path.join(output_directory, "Violet_Transformed_Data.npy"))
    print("Violet Loadings", np.shape(violet_loadings))

    # Load Mask
    indicies, image_height, image_width = load_mask(output_directory)

    sample_size = 1000
    violet_sample = violet_loadings[:, 0:sample_size]

    # Reconstruct Violet Data
    reconstructed_matrix = np.dot(np.transpose(blue_components), violet_sample)
    reconstructed_matrix = np.transpose(reconstructed_matrix)

    figure_1 = plt.figure()
    rows = 1
    columns = 3

    plt.ion()
    for timepoint in range(1000):

        #original_frame = original_violet_data[timepoint]
        reconstructed_frame = reconstructed_matrix[timepoint]
        #difference = np.subtract(original_frame, reconstructed_frame)

        #original_axis       = figure_1.add_subplot(rows, columns, 1)
        reconstructed_axis  = figure_1.add_subplot(rows, columns, 2)
        #difference_axis     = figure_1.add_subplot(rows, columns, 3)

        reconstructed_image = Widefield_General_Functions.create_image_from_data(reconstructed_frame, indicies, image_height, image_width)


        #original_axis.imshow(original_frame, vmin=-1, vmax=1, cmap='jet')
        reconstructed_axis.imshow(reconstructed_image, cmap='Greys_r')
        #difference_axis.imshow(np.abs(difference), vmin=0, vmax=1, cmap='jet')
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    print(np.shape(reconstructed_matrix))


def hemodynamic_correction(U, SVT_470, SVT_405,
                            fs=30.,
                           freq_lowpass=14.,
                           freq_highpass=0.1,
                           nchunks=200):

    #nchunks=1024

    # split channels and subtract the mean to each
    SVTa = SVT_470  # [:,0::2]
    SVTb = SVT_405  # [:,1::2]

    # reshape U
    dims = U.shape
    U = U.reshape([-1, dims[-1]])


    # Single channel sampling rate
    fs = fs
    # Highpass filter
    if not freq_highpass is None:
        SVTa = highpass(SVTa, w=freq_highpass, fs=fs)
        SVTb = highpass(SVTb, w=freq_highpass, fs=fs)
    if not freq_lowpass is None:
        if freq_lowpass < fs / 2:
            SVTb = lowpass(SVTb, freq_lowpass, fs=fs)
            SVTa = lowpass(SVTa, freq_lowpass, fs=fs) # I Added This In
        else:
            print('Skipping lowpass on the violet channel.')

    # subtract the mean
    SVTa = (SVTa.T - np.nanmean(SVTa, axis=1)).T.astype('float32')
    SVTb = (SVTb.T - np.nanmean(SVTb, axis=1)).T.astype('float32')

    npix = U.shape[0]
    idx = np.array_split(np.arange(0, npix), nchunks)

    # find the coefficients
    rcoeffs = np.zeros((npix))
    for i, ind in tqdm(enumerate(idx)):
        a = np.dot(U[ind, :], SVTa)
        b = np.dot(U[ind, :], SVTb)
        rcoeffs[ind] = np.sum(a * b, axis=1) / np.sum(b * b, axis=1)

    # drop nan
    rcoeffs[np.isnan(rcoeffs)] = 1.e-10

    # find the transformation
    T = np.dot(np.linalg.pinv(U), (U.T * rcoeffs).T)

    # apply correction
    SVTcorr = SVTa - np.dot(T, SVTb)

    # return a zero mean SVT
    SVTcorr = (SVTcorr.T - np.nanmean(SVTcorr, axis=1)).T.astype('float32')

    # put U dims back in case its used sequentially
    U = U.reshape(dims)

    print("SVT Corr Shape", np.shape(SVTcorr))
    print("R Coefs Shape", np.shape(rcoeffs))
    print("T", np.shape(T))

    return SVTcorr.astype('float32'), rcoeffs.astype('float32'), T.astype('float32')


def perform_heamocorrection(base_directory):

    #blue_components = np.load(os.path.join(output_directory, "Blue_Data_SVD_incremental_components_200.npy"))
    #blue_transformed_data = np.load(os.path.join(output_directory, "Blue_Data_SVD_incremental_transformed_data_200.npy"))

    svt = np.load(os.path.join(base_directory, "SVT_Churchland_Aprox.npy"))
    u = np.load(os.path.join(base_directory, "U_Churchland_Aprox.npy"))

    number_of_components, number_of_frames = np.shape(svt)
    print("NUmber of component", number_of_components)
    print("Number of frames", number_of_frames)

    blue_indicies = list(range(0, number_of_frames, 2))
    violet_indicies = list(range(1, number_of_frames, 2))

    blue_svt = svt[:, blue_indicies]
    violet_svt = svt[:, violet_indicies]

    print("Blue SVT", np.shape(blue_svt))
    print("VIolet SVT", np.shape(violet_svt))

    SVTcorr, rcoeffs, T = hemodynamic_correction(u, blue_svt, violet_svt)

    return SVTcorr


def subtract_mean(ground_truth_data):

    print(np.shape(ground_truth_data))

    ground_truth_mean = np.mean(ground_truth_data, axis=0)
    ground_truth_data = np.subtract(ground_truth_data, ground_truth_mean)

    return ground_truth_data


def get_delta_f(data_sample):

    # Get Video Structure
    width, height, frames = np.shape(data_sample)

    # Flatten Video
    data_sample = np.reshape(data_sample, (width * height, frames))

    # Transpose So Samples Are Along The First Axis
    data_sample = np.transpose(data_sample)

    # Get Baseline
    baseline = np.percentile(a=data_sample, q=5, axis=0)

    # Subtract Baseline
    delta_f = np.subtract(data_sample, baseline)

    # Clip Negative Values
    delta_f = np.clip(delta_f, a_min=0, a_max=None)

    # Divide By Baseline
    delta_f_over_f = np.divide(delta_f, baseline)

    # Put Back Into Original Structure
    delta_f_over_f = np.transpose(delta_f_over_f)
    delta_f_over_f = np.reshape(delta_f_over_f, (width, height, frames))

    return delta_f_over_f


def normalise_sample(data_sample):

    # Get Video Structure
    width, height, frames = np.shape(data_sample)

    # Flatten Video
    data_sample = np.reshape(data_sample, (width * height, frames))

    # Transpose So Samples Are Along The First Axis
    data_sample = np.transpose(data_sample)

    # Subtract Baseline
    baseline = np.min(data_sample, axis=0)
    data_sample = np.subtract(data_sample, baseline)

    # Divide By Max
    max_vector = np.max(data_sample, axis=0)
    data_sample = np.divide(data_sample, max_vector)

    # Put Back Into Original Shape
    data_sample = np.transpose(data_sample)
    data_sample = np.reshape(data_sample, (width, height, frames))

    return data_sample


def view_heamocorrected_data(base_directory, sample_size=2000):

    # Load Mask
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Load Blue Components
    blue_components = np.load(os.path.join(base_directory, "U_Churchland_Aprox.npy"))

    # Load Corrected Data
    corrected_data = np.load(os.path.join(output_directory, "SVT_Corr.npy"))
    print("Blue Components", np.shape(blue_components))
    print(np.shape(corrected_data))
    corrected_sample = corrected_data[:, 0:sample_size]
    reconstructed_data = np.dot(blue_components, corrected_sample)
    print("Reconstructed Data Shape", np.shape(reconstructed_data))

    # Get Delta F
    #reconstructed_data = get_delta_f(reconstructed_data)

    # Normalise Delta F Over F
    reconstructed_data = normalise_sample(reconstructed_data)

    sample_min = np.min(reconstructed_data)
    sample_max = np.max(reconstructed_data)

    sample_min = np.percentile(reconstructed_data, q=5)
    sample_max = np.percentile(reconstructed_data, q=99)

    print("sample min", sample_min)
    print("sample max", sample_max)

    print(np.shape(reconstructed_data))
    plt.ion()
    for frame_index in range(sample_size-1):

        frame_data = reconstructed_data[:, :, frame_index:frame_index+1]
        frame_data = np.mean(frame_data, axis=2)
        frame_data = np.transpose(frame_data)

        frame_data = np.multiply(frame_data, mask)

        frame_data = ndimage.gaussian_filter(frame_data, sigma=2)


        plt.imshow(frame_data, cmap='plasma', vmin=0, vmax=1)
        plt.draw()
        plt.pause(0.001)
        plt.clf()





output_directory = "/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"


# Perform Blue SVD
#view_blue_components(output_directory)

# Decompose Violet Matrix
#decompose_violet_video(output_directory)

#view_violet_reconstruction(output_directory)

# Perform Heamocorrection
SVTcorr = perform_heamocorrection(output_directory)
np.save(os.path.join(output_directory, "SVT_Corr.npy"), SVTcorr)


view_heamocorrected_data(output_directory)
