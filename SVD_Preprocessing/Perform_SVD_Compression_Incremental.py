import numpy as np
import os
import h5py
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
#from dask_ml.decomposition import IncrementalPCA

import datetime
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def perform_svd_compression(base_directory):

    # Set File Settings
    data_file = os.path.join(base_directory, "Delta_F.hdf5")


    # Load Delta F Data
    downsampled_file_object = h5py.File(data_file, 'r')
    data_matrix = downsampled_file_object["Data"]
    number_of_frames, number_of_pixels = np.shape(data_matrix)

    # Get Chunk Structure
    chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(chunk_size, number_of_frames)

    # Create Model
    model = IncrementalPCA(n_components=500)

    for chunk_index in range(number_of_chunks):
        print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.datetime.now())
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]

        model.partial_fit(chunk_data)

    components = model.components_
    svd_mean = model.mean_

    np.save(os.path.join(base_directory, "SVD_Components.npy"), components)
    np.save(os.path.join(base_directory, "SVD_Mean.npy"), svd_mean)

    # Transform_Data
    transformed_data = []
    for chunk_index in range(number_of_chunks):
        chunk_start = chunk_starts[chunk_index]
        chunk_stop = chunk_stops[chunk_index]
        chunk_data = data_matrix[chunk_start:chunk_stop]
        transformed_chunk = model.transform(chunk_data)
        for datapoint in transformed_chunk:
            transformed_data.append(datapoint)

    transformed_data = np.array(transformed_data)
    print("Transformed Data Shape", np.shape(transformed_data))
    np.save(os.path.join(base_directory, "SVD_Transformed_data.npy"), transformed_data)


