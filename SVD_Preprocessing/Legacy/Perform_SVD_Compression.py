import numpy as np
import os
import h5py
import sys
from sklearn.decomposition import TruncatedSVD

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def perform_svd_compression(base_directory, data_file, output_file):

    # Load Delta F Data
    downsampled_file_object = h5py.File(os.path.join(base_directory, data_file), 'r')
    data_matrix = downsampled_file_object["Data"]
    data_matrix = np.array(data_matrix[:, 0:10000])
    print(data_matrix.nbytes)

    data_matrix = np.transpose(data_matrix)
    print(data_matrix.nbytes)

    model = TruncatedSVD(n_components=50)
    transformed_data = model.fit_transform(data_matrix)
    components = model.components_
    singular_values = model.singular_values_

    np.save(os.path.join(base_directory, output_file) + "_transformed_data.npy", transformed_data)
    np.save(os.path.join(base_directory, output_file) + "_components.npy", components)
    np.save(os.path.join(base_directory, output_file) + "_singular_values.npy", singular_values)


base_directory = r"/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
perform_svd_compression(base_directory, "Masked_Blue_Data.hdf5", "Blue_Data_SVD")