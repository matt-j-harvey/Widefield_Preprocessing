import numpy as np
import sklearn.svm
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import h5py

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_generous_mask(home_directory):

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



def ResampleLinear1D(original, targetLen):

    original = np.array(original, dtype=float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=float)
    index_floor = np.array(index_arr, dtype=int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp



def downsample_ai_traces(base_directory, sanity_check=True):

    # Load Frame Times
    frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    frame_times = Widefield_General_Functions.invert_dictionary(frame_times)

    # Load AI Recorder File
    ai_filename = Widefield_General_Functions.get_ai_filename(base_directory)
    ai_data = Widefield_General_Functions.load_ai_recorder_file(base_directory + "/" + ai_filename)

    # Extract Relevant Traces
    stimuli_dictionary = Widefield_General_Functions.create_stimuli_dictionary()
    running_trace = ai_data[stimuli_dictionary["Running"]]
    lick_trace = ai_data[stimuli_dictionary["Lick"]]

    # Load Delta F Matrix
    delta_f_matrix_filepath = os.path.join(base_directory, "Delta_F.hdf5")
    delta_f_matrix_container = h5py.File(delta_f_matrix_filepath, 'r')
    delta_f_matrix = delta_f_matrix_container['Data']

    # Get Data Structure
    number_of_timepoints = np.shape(delta_f_matrix)[0]
    imaging_start = frame_times[0]
    imaging_stop = frame_times[number_of_timepoints - 1]

    # Get Traces Only While Imaging
    imaging_running_trace = running_trace[imaging_start:imaging_stop]
    imaging_lick_trace = lick_trace[imaging_start:imaging_stop]

    # Downsample Traces
    downsampled_running_trace = ResampleLinear1D(imaging_running_trace, number_of_timepoints)
    downsampled_lick_trace = ResampleLinear1D(imaging_lick_trace, number_of_timepoints)

    return downsampled_running_trace, downsampled_lick_trace



def visualise_coefficients(base_directory, coefficients):

    indicies, image_height, image_width = load_generous_mask(base_directory)

    number_of_dimensions = np.ndim(coefficients)

    if number_of_dimensions == 1:
        image = Widefield_General_Functions.create_image_from_data(coefficients, indicies, image_height, image_width)
        plt.imshow(image)
        plt.show()

    elif number_of_dimensions == 2:

        dim_1, dim_2 = np.shape(coefficients)

        if dim_1 > dim_2:
            coefficients = np.transpose(coefficients)

        nuber_of_samples = np.shape(coefficients)[0]
        plt.ion()

        for x in range(nuber_of_samples):
            image = Widefield_General_Functions.create_image_from_data(coefficients[x], indicies, image_height, image_width)
            plt.title(str(x))
            plt.imshow(image)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        plt.ioff()
    plt.close()




session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"

]

for session_index in range(len(session_list)):

    print("Session: ", session_index, " of ", len(session_list))

    # Select Session Directory
    base_directory = session_list[session_index]

    # Create Output Directory
    output_directory = os.path.join(base_directory, "Interpretable_Regressors")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)


    # Load Behavioural Data
    downsampled_running_trace, downsampled_lick_trace = downsample_ai_traces(base_directory)

    sample_size = 10000

    # Load Data
    data_file = os.path.join(base_directory, "Delta_F.hdf5")
    data_container = h5py.File(data_file, 'r')
    delta_f = data_container['Data']


    # Get Sample Data
    running_sample = downsampled_running_trace[0:sample_size+9]
    running_sample = moving_average(running_sample, 10)
    print(len(running_sample))
    #plt.plot(running_sample)

    #running_sample = np.diff(running_sample)

    #plt.plot(running_sample)
    #plt.show()
    lick_sample = downsampled_lick_trace[0:sample_size]

    delta_f_sample = delta_f[0:sample_size]
    delta_f_sample = np.nan_to_num(delta_f_sample)
    design_matrix = np.array([running_sample, lick_sample])
    design_matrix = np.transpose(design_matrix)

    model = Ridge()
    model.fit(X=design_matrix, y=delta_f_sample)

    coef_matrix = model.coef_
    coef_matrix = np.transpose(coef_matrix)
    print("Ceof matrix", np.shape(coef_matrix))
    running_coefs = coef_matrix[0]
    lick_coefs = coef_matrix[1]

    #visualise_coefficients(base_directory, running_coefs)
    #visualise_coefficients(base_directory, lick_coefs)

    np.save(os.path.join(output_directory, "Lick_Regressor.npy"), lick_coefs)
    np.save(os.path.join(output_directory, "Running_Regressor.npy"), lick_coefs)

    print("Design Matrix", np.shape(design_matrix))
    print("Delta F Sample", np.shape(delta_f_sample))



