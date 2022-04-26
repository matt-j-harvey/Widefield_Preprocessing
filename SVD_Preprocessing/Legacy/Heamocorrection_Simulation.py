import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

def gaussian_kernel(height, width,sigma=0.5):
    """
      2D gaussian mask - should give the same result as MATLAB's
      fspecial('gaussian',[shape],[sigma])
      """

    shape = (height, width)
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    h = np.divide(h, np.max(h))

    return h


def create_video(kernel, loading_trace):

    video_height, video_width = np.shape(kernel)
    number_of_timepoints = len(loading_trace)
    video_matrix = np.zeros((number_of_timepoints, video_height, video_width))

    for timepoint in range(number_of_timepoints):
        loading_value = loading_trace[timepoint]
        video_matrix[timepoint] = np.multiply(kernel, loading_value)

    return video_matrix



def get_random_walk(length):

    y = 0
    result = []
    for timepoint in range(length):
        result.append(y)
        y += np.random.normal(scale=1)

    result = np.divide(result, np.max(np.abs(result)))
    return np.array(result)



def generate_pseudo_movie(number_of_timepoints, video_height, video_width, number_of_components=20):

    video_data = []

    for component_index in range(number_of_components):
        kernel = gaussian_kernel(video_height, video_width, sigma=np.random.randint(5, 20))
        kernel = np.roll(kernel, axis=0, shift=np.random.randint(-video_height, video_height))
        kernel = np.roll(kernel, axis=1, shift=np.random.randint(-video_width, video_width))
        timeseries = get_random_walk(number_of_timepoints)
        video = create_video(kernel, timeseries)
        video_data.append(video)

    video_data = np.array(video_data)
    video_data = np.sum(video_data, axis=0)

    return video_data


def generate_data(output_directory):

    number_of_timepoints = 1000
    video_height = 400
    video_width = 500
    timepoints = list(range(number_of_timepoints))

    """
    # Create Fake Violet Changes
    violet_kernel = gaussian_kernel(video_height, video_width, sigma=100)
    violet_trace = get_random_walk(number_of_timepoints)
    violet_video = create_video(violet_kernel, violet_trace)

    # Create Fake Blue Changes
    blue_kernel_1 = gaussian_kernel(video_height, video_width, sigma=20)
    blue_kernel_1 = np.roll(blue_kernel_1, axis=0, shift=-100)
    blue_kernel_1 = np.roll(blue_kernel_1, axis=1, shift=100)
    blue_trace_1 = get_random_walk(number_of_timepoints)

    blue_kernel_2 = gaussian_kernel(video_height, video_width, sigma=20)
    blue_kernel_2 = np.roll(blue_kernel_2, axis=0, shift=-100)
    blue_kernel_2 = np.roll(blue_kernel_2, axis=1, shift=-100)
    blue_trace_2 = get_random_walk(number_of_timepoints)

    blue_movie_1 = create_video(blue_kernel_1, blue_trace_1)
    blue_movie_2 = create_video(blue_kernel_2, blue_trace_2)
    blue_movie = np.add(blue_movie_1, blue_movie_2)
    """

    violet_video = generate_pseudo_movie(number_of_timepoints, video_height, video_width)
    blue_video = generate_pseudo_movie(number_of_timepoints, video_height, video_width)
    final_video = np.add(violet_video, blue_video)

    # Create Noise
    noise = np.random.uniform(low=-0.2, high=0.2, size=(number_of_timepoints, video_height, video_width))
    final_video = np.add(final_video, noise)
    violet_video_noise = np.add(violet_video, noise)


    """
    plt.ion()
    for frame in final_video:
        plt.imshow(frame, vmin=np.min(final_video), vmax=np.max(final_video))
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    """

    np.save(os.path.join(output_directory, "Blue_Video_Groundtruth.npy"), blue_video)
    np.save(os.path.join(output_directory, "Violet_Video_Groundtruth.npy"), violet_video)
    np.save(os.path.join(output_directory, "Blue_Video.npy"), final_video)
    np.save(os.path.join(output_directory, "Violet_Video.npy"), violet_video_noise)


def compute_blue_svd(output_directory):

    # Compute Blue SVD
    blue_data = np.load(os.path.join(output_directory, "Blue_Video.npy"))
    print(np.shape(blue_data))

    number_of_timepoints, image_height, image_width = np.shape(blue_data)
    blue_data = np.reshape(blue_data, (number_of_timepoints, image_height * image_width))
    model = TruncatedSVD(n_components=10)

    transformed_data = model.fit_transform(blue_data)
    components = model.components_
    singular_values = model.singular_values_

    np.save(os.path.join(output_directory, "Blue_Components.npy"), components)
    np.save(os.path.join(output_directory, "Blue_Transformed_Data.npy"), transformed_data)
    np.save(os.path.join(output_directory, "Blue_Singular_Values.npy"), singular_values)


def view_blue_components(output_directory):

    blue_components = np.load(os.path.join(output_directory, "Blue_Components.npy"))
    number_of_components = np.shape(blue_components)[0]

    figure_1 = plt.figure()
    rows = 1
    columns = number_of_components
    for x in range(number_of_components):
        axis = figure_1.add_subplot(rows, columns, x+1)

        component_data = blue_components[x]
        component_data = np.ndarray.reshape(component_data, (400, 500))
        axis.imshow(component_data)

    plt.show()


def decompose_violet_video(output_directory):

    # Load Violet Video
    violet_video = np.load(os.path.join(output_directory, "Violet_Video.npy"))

    # Reshape Violet Video
    number_of_frames, image_height, image_width = np.shape(violet_video)
    violet_video = np.reshape(violet_video, (number_of_frames, (image_height * image_width)))

    # Load Blue Components
    blue_components = np.load(os.path.join(output_directory, "Blue_Components.npy"))

    # Invert Blue Components
    inverse_components = np.linalg.pinv(blue_components)

    # Get New Violet Loadings
    new_v = np.dot(np.transpose(inverse_components), np.transpose(violet_video))

    # Save These
    np.save(os.path.join(output_directory, "Violet_Transformed_Data.npy"), new_v)


def view_violet_reconstruction(output_directory):

    # Load Blue Components
    blue_components = np.load(os.path.join(output_directory, "Blue_Components.npy"))

    # Load Violet Loadings
    violet_loadings = np.load(os.path.join(output_directory, "Violet_Transformed_Data.npy"))

    number_of_frames = 1000
    image_height = 400
    image_width = 500

    # Reconstruct Violet Data
    reconstructed_matrix = np.dot(np.transpose(blue_components), violet_loadings)
    reconstructed_matrix = np.transpose(reconstructed_matrix)
    reconstructed_matrix = np.reshape(reconstructed_matrix, (number_of_frames, image_height, image_width))

    # Load Original Violet Data
    original_violet_data = np.load(os.path.join(output_directory, "Violet_Video.npy"))

    figure_1 = plt.figure()
    rows = 1
    columns = 3

    plt.ion()
    for timepoint in range(1000):

        original_frame = original_violet_data[timepoint]
        reconstructed_frame = reconstructed_matrix[timepoint]
        difference = np.subtract(original_frame, reconstructed_frame)

        original_axis       = figure_1.add_subplot(rows, columns, 1)
        reconstructed_axis  = figure_1.add_subplot(rows, columns, 2)
        difference_axis     = figure_1.add_subplot(rows, columns, 3)

        original_axis.imshow(original_frame, vmin=-1, vmax=1, cmap='jet')
        reconstructed_axis.imshow(reconstructed_frame, vmin=-1, vmax=1, cmap='jet')
        difference_axis.imshow(np.abs(difference), vmin=0, vmax=1, cmap='jet')
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    print(np.shape(reconstructed_matrix))


def hemodynamic_correction(U, SVT_470, SVT_405,
                           # fs=30.,
                           # freq_lowpass=14.,
                           # freq_highpass=0.1,
                           nchunks=1024):
    # split channels and subtract the mean to each
    SVTa = SVT_470  # [:,0::2]
    SVTb = SVT_405  # [:,1::2]

    # reshape U
    dims = U.shape
    U = U.reshape([-1, dims[-1]])

    """
    # Single channel sampling rate
    fs = fs
    # Highpass filter
    if not freq_highpass is None:
        SVTa = highpass(SVTa, w=freq_highpass, fs=fs)
        SVTb = highpass(SVTb, w=freq_highpass, fs=fs)
    if not freq_lowpass is None:
        if freq_lowpass < fs / 2:
            SVTb = lowpass(SVTb, freq_lowpass, fs=fs)
        else:
            print('Skipping lowpass on the violet channel.')
    """
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


def perform_heamocorrection(output_directory):

    blue_components = np.load(os.path.join(output_directory, "Blue_Components.npy"))

    blue_singular_values = np.load(os.path.join(output_directory, "Blue_Singular_Values.npy"))
    blue_transformed_data = np.load(os.path.join(output_directory, "Blue_Transformed_Data.npy"))
    #blue_transformed_data = np.multiply(blue_singular_values, blue_transformed_data_unit)

    violet_transformed_data = np.load(os.path.join(output_directory, "Violet_Transformed_Data.npy"))

    blue_transformed_data = np.transpose(blue_transformed_data)
    # violet_transformed_data = np.transpose(violet_transformed_data)

    blue_components = np.transpose(blue_components)
    print("Blue Components", np.shape(blue_components))
    print("Blue Transformed Data", np.shape(blue_transformed_data))
    print("Violet Transformed Data", np.shape(violet_transformed_data))

    SVTcorr, rcoeffs, T = hemodynamic_correction(blue_components, blue_transformed_data, violet_transformed_data)

    return SVTcorr


def subtract_mean(ground_truth_data):

    print(np.shape(ground_truth_data))

    ground_truth_mean = np.mean(ground_truth_data, axis=0)
    ground_truth_data = np.subtract(ground_truth_data, ground_truth_mean)

    return ground_truth_data


output_directory = "/media/matthew/29D46574463D2856/Heamocorrection_Simulation"

# Generate Data
generate_data(output_directory)

# Perform Blue SVD
compute_blue_svd(output_directory)
view_blue_components(output_directory)

# Decompose Violet Matrix
decompose_violet_video(output_directory)
view_violet_reconstruction(output_directory)

# Perform Heamocorrection
SVTcorr = perform_heamocorrection(output_directory)

# View Heamocorrected Data
blue_components = np.load(os.path.join(output_directory, "Blue_Components.npy"))
blue_components = np.transpose(blue_components)
reconstructed_corrected_movie = np.dot(blue_components, SVTcorr)
print("Reconstructed Corrected Movie", np.shape(reconstructed_corrected_movie))
reconstructed_corrected_movie = np.transpose(reconstructed_corrected_movie)
reconstructed_corrected_movie = np.divide(reconstructed_corrected_movie, np.max(np.abs(reconstructed_corrected_movie)))

# Load Ground Truth Data
ground_truth_blue_video = np.load(os.path.join(output_directory, "Blue_Video_Groundtruth.npy"))
ground_truth_blue_video = subtract_mean(ground_truth_blue_video)
ground_truth_blue_video = np.divide(ground_truth_blue_video, np.max(np.abs(ground_truth_blue_video)))

# Subtract Ground Truth Mean
print("Ground truth shape", np.shape(ground_truth_blue_video))


figure_1 = plt.figure()
rows = 1
columns = 2

plt.ion()
number_of_timepoints = 1000

for timepoint_index in range(number_of_timepoints):

    # Get Reconstrcuted Frame
    reconstructed_frame = reconstructed_corrected_movie[timepoint_index]
    reconstructed_frame_image = np.reshape(reconstructed_frame, (400, 500))

    # Get Ground Truth Frame
    ground_truth_frame = ground_truth_blue_video[timepoint_index]

    reconstruction_axis = figure_1.add_subplot(rows, columns, 1)
    ground_truth_axis =figure_1.add_subplot(rows, columns, 2)

    reconstruction_axis.imshow(reconstructed_frame_image, vmin=-1, vmax=1)
    ground_truth_axis.imshow(ground_truth_frame, vmin=-1, vmax=1)
    plt.draw()
    plt.pause(0.001)
    plt.clf()
