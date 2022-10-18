#  wfield - tools to analyse widefield data - motion correction
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import cv2

import numpy as np
from skimage.transform import AffineTransform
import tables
from glob import glob
from os.path import join as pjoin
from datetime import datetime
from skimage.transform import warp
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.interpolate import interp1d
from scipy.sparse import load_npz, issparse,csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import sys

import Preprocessing_Utils

cv2.setNumThreads(10)


def get_blue_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Blue_Data" in file:
            return base_directory + "/" + file

def get_violet_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Violet_Data" in file:
            return base_directory + "/" + file

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"


def runpar(f,X,nprocesses = None,**kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = parinit, processes=nprocesses) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res




def findTransformECC(template, dst, M, warp_mode, criteria, inputMask, gaussFiltSize):
    return cv2.findTransformECC(template, dst,
                                M, warp_mode,
                                criteria,
                                inputMask=inputMask,
                                gaussFiltSize=gaussFiltSize)


cv2ver = cv2.__version__.split('.')
if (int(cv2ver[0]) == 3) and (int(cv2ver[1]) <= 4):
    if int(cv2ver[2]) <= 5:
        def findTransformECC(template,
                             dst,
                             M,
                             warp_mode,
                             criteria,
                             inputMask,
                             gaussFiltSize):
            return cv2.findTransformECC(template, dst,
                                        M, warp_mode,
                                        criteria,
                                        inputMask=inputMask)
elif (int(cv2ver[0]) == 4) and (int(cv2ver[1]) <= 1):
    # gaussFiltSize is a mandatory input on opencv 4.4 but not 4.1
    def findTransformECC(template,
                         dst,
                         M,
                         warp_mode,
                         criteria,
                         inputMask,
                         gaussFiltSize):
        return cv2.findTransformECC(template, dst,
                                    M, warp_mode,
                                    criteria,
                                    inputMask=inputMask)


def registration_ecc(frame, template,
                     niter=100,
                     eps0=1e-3,
                     warp_mode=cv2.MOTION_EUCLIDEAN,
                     prepare=True,
                     gaussian_filter=1,
                     hann=None,
                     **kwargs):
    h, w = template.shape
    if hann is None:
        hann = cv2.createHanningWindow((w, h), cv2.CV_32FC1)
        hann = (hann * 255).astype('uint8')
    dst = frame.astype('float32')
    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                niter, eps0)
    (res, M) = findTransformECC(template, dst,
                                M, warp_mode,
                                criteria,
                                inputMask=hann, gaussFiltSize=gaussian_filter)

    dst = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return M, np.clip(dst, 0, (2 ** 16 - 1)).astype('uint16')


def _xy_rot_from_affine(affines):
    '''
    helper function to parse affine parameters from ECC
    '''
    xy = []
    rot = []
    for r in affines:
        M = np.vstack([r, np.array([0, 0, 1])])
        M = AffineTransform(M)
        xy.append(M.translation)
        rot.append(M.rotation)
    rot = np.rad2deg(np.array(rot))
    xy = np.array(xy)
    return xy, rot


def registration_upsample(frame, template):
    h, w = frame.shape
    dst = frame.astype('float32')
    (xs, ys), sf = cv2.phaseCorrelate(dst, template.astype('float32'))
    M = np.float32([[1, 0, xs], [0, 1, ys]])
    dst = cv2.warpAffine(dst, M, (w, h))
    return (xs, ys), (np.clip(dst, 0, (2 ** 16 - 1))).astype('uint16')


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


def _register_multichannel_stack(frames, templates, mode='2d',
                                 niter=100,
                                 eps0=1e-3,
                                 warp_mode=cv2.MOTION_EUCLIDEAN):  # mode 2d

    nframes, nchannels, h, w = frames.shape
    if mode == 'ecc':
        hann = cv2.createHanningWindow((w, h), cv2.CV_32FC1)
        hann = (hann * 255).astype('uint8')

    ys = np.zeros((nframes, nchannels), dtype=np.float32)
    xs = np.zeros((nframes, nchannels), dtype=np.float32)
    rot = np.zeros((nframes, nchannels), dtype=np.float32)
    stack = np.zeros_like(frames, dtype='uint16')

    for ichan in range(nchannels):
        chunk = frames[:, ichan].squeeze()
        if mode == '2d':
            res = runpar(registration_upsample, chunk,
                         template=templates[ichan])
            ys[:, ichan] = np.array([r[0][1] for r in res], dtype='float32')
            xs[:, ichan] = np.array([r[0][0] for r in res], dtype='float32')

        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk,
                         template=templates[ichan],
                         hann=hann,
                         niter=niter,
                         eps0=eps0,
                         warp_mode=warp_mode)


            xy, rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:, ichan] = xy[:, 1]
            xs[:, ichan] = xy[:, 0]
            rot[:, ichan] = rots
        stack[:, ichan, :, :] = np.stack([r[1] for r in res])
    return (xs, ys, rot), stack






def get_reference_images(blue_matrix, violet_matrix, reference_size=60, start=1000, image_height=600, image_width=608, mode='ecc'):

    # Extract Chunk Data
    combined_data = np.array([blue_matrix[:, start:start + reference_size], violet_matrix[:, start:start + reference_size]])

    # Reshape Combined Data
    channels, pixels, frames = np.shape(combined_data)
    combined_data = combined_data.reshape(channels, image_height, image_width, frames)
    combined_data = np.moveaxis(combined_data, [0, 1, 2, 3], [1, 2, 3, 0])

    # Align Frames Within This Chunk
    refs = combined_data[0].astype('float32')
    _, refs = _register_multichannel_stack(combined_data, refs, mode=mode)

    # Take The Mean
    refs = np.mean(refs, axis=0).astype('float32')

    return refs




def plot_registration_shifts(base_directory):
    print("Plotting registration", base_directory)

    # Load Data
    x_shifts = np.load(os.path.join(base_directory, "X_Shifts.npy"))
    y_shifts = np.load(os.path.join(base_directory, "Y_Shifts.npy"))
    r_shifts = np.load(os.path.join(base_directory, "R_Shifts.npy"))

    # Create Figure
    figure_1 = plt.figure()
    rows = 2
    columns = 1
    translation_axis = figure_1.add_subplot(rows, columns, 1)
    rotation_axis = figure_1.add_subplot(rows, columns, 2)

    # Plot Data
    translation_axis.plot(x_shifts, c='b')
    translation_axis.plot(y_shifts, c='r')
    rotation_axis.plot(r_shifts, c='g')

    # Save Figure
    plt.savefig(os.path.join(base_directory, "Motion_Correction_Shifts.png"))
    plt.close()


def perform_motion_correction(base_directory, output_directory, output_file="Motion_Corrected_Mask_Data.hdf5"):

    # Get Blue and Violet Files
    blue_file                   = get_blue_file(base_directory)
    violet_file                 = get_violet_file(base_directory)

    # Load Mask
    indicies, image_height, image_width = load_generous_mask(output_directory)


    # Load Data
    blue_data_container = h5py.File(blue_file, 'r')
    violet_data_container = h5py.File(violet_file, 'r')
    blue_data = blue_data_container["Data"]
    violet_data = violet_data_container["Data"]

    # Get Reference Images
    reference_images = get_reference_images(blue_data, violet_data)

    # Get Chunk Structure
    number_of_pixels, number_of_frames = np.shape(blue_data)
    number_of_active_pixels = len(indicies)
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_frames)

    x_shifts = []
    y_shifts = []
    r_shifts = []

    # Process Data
    #file_cache_size = 16561440000
    with h5py.File(os.path.join(base_directory, output_file), "w") as f:
        corrected_blue_dataset = f.create_dataset("Blue_Data", (number_of_active_pixels, number_of_frames), dtype=np.uint16, chunks=True, compression="gzip")
        corrected_violet_dataset = f.create_dataset("Violet_Data", (number_of_active_pixels, number_of_frames), dtype=np.uint16, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk Index", chunk_index, " of ", number_of_chunks, "Time: ", datetime.now())
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])
            chunk_size = chunk_sizes[chunk_index]

            # Load Chunk Data
            combined_data = np.array([blue_data[:, chunk_start:chunk_stop], violet_data[:, chunk_start:chunk_stop]])

            # Reshape Combined Data
            channels, pixels, frames = np.shape(combined_data)
            combined_data = combined_data.reshape(channels, image_height, image_width, frames)
            combined_data = np.moveaxis(combined_data, [0, 1, 2, 3], [1, 2, 3, 0])

            # Perform Motion Correction
            (xs, ys, rot), corrected = _register_multichannel_stack(combined_data, reference_images, mode='ecc')

            # Record The Shifts
            x_shifts.append(xs)
            y_shifts.append(ys)
            r_shifts.append(rot)

            # Reshape The Corrected Data
            combined_data = None
            corrected_blue = corrected[:, 0]
            corrected_violet = corrected[:, 1]
            corrected_blue = np.reshape(corrected_blue, (chunk_size, image_height * image_width))
            corrected_violet = np.reshape(corrected_violet, (chunk_size, image_height * image_width))

            # Select Only The Masked Pixels
            corrected_blue = corrected_blue[:, indicies]
            corrected_violet = corrected_violet[:, indicies]

            # Put Back
            corrected_blue_dataset[:, chunk_start:chunk_stop] = np.transpose(corrected_blue)
            corrected_violet_dataset[:, chunk_start:chunk_stop] = np.transpose(corrected_violet)

    x_shifts = np.concatenate(x_shifts)
    y_shifts = np.concatenate(y_shifts)
    r_shifts = np.concatenate(r_shifts)

    np.save(os.path.join(output_directory, "X_Shifts.npy"), x_shifts)
    np.save(os.path.join(output_directory, "Y_Shifts.npy"), y_shifts)
    np.save(os.path.join(output_directory, "R_Shifts.npy"), r_shifts)

    # Plot Registration Shifts
    plot_registration_shifts(output_directory)