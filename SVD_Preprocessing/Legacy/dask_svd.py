import dask
import dask.array as da
from dask_ml.decomposition import TruncatedSVD
from dask.distributed import Client
import h5py
import os
import numpy as np
import datetime

#client = Client()


base_directory = r"/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
data_file = "Masked_Blue_Data.hdf5"
data_file_object = h5py.File(os.path.join(base_directory, data_file), 'r')
data_matrix = data_file_object['Data']
print("Data Matrix Shape", np.shape(data_matrix))
#data_matrix_dask = da.from_array(data_file_object['/Data'], chunks=(5000, 218588))
data_matrix_dask = da.from_array(data_file_object['/Data'], chunks=(10000, 218588))
#data_matrix_dask = da.from_array(data_file_object['/Data'])


print("Starting:", datetime.datetime.now())
model = TruncatedSVD(n_components=200, compute=True)
model.fit(data_matrix_dask)

"""
u, s, v = da.linalg.svd_compressed(data_matrix_dask, 200, n_power_iter=2)
cu,cs,cv=da.compute(u,s,v)
print("Finished:", datetime.datetime.now())
"""


output_file = "Blue_Data_SVD"


np.save(os.path.join(base_directory, output_file) + "_dask_components.npy", model.components_)
np.save(os.path.join(base_directory, output_file) + "_dask_singular_values.npy", model.singular_values_)
np.save(os.path.join(base_directory, output_file) + "_dask_transformed_data.npy", model.transform(data_matrix_dask))

"""
np.save(os.path.join(base_directory, output_file) + "_dask_components.npy", u)
np.save(os.path.join(base_directory, output_file) + "_dask_singular_values.npy", s)
np.save(os.path.join(base_directory, output_file) + "_dask_transformed_data.npy", v)
"""
