from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import h5py
import tables
from tqdm import tqdm
from numpy.linalg import svd


def get_trial_baseline(idx,frames_average,onsets):
    if len(frames_average.shape) <= 3:
        return frames_average
    else:
        if onsets is None:
            print(' Trial onsets not defined, using the first trial')
            return frames_average[0]
        return frames_average[np.where(onsets<=idx)[0][-1]]


def make_overlapping_blocks(dims, blocksize=128, overlap=16):
    '''
    Creates overlapping block indices to span an image
    '''

    w, h = dims
    blocks = []
    for i, a in enumerate(range(0, w, blocksize - overlap)):
        for j, b in enumerate(range(0, h, blocksize - overlap)):
            blocks.append([(a, np.clip(a + blocksize, 0, w)), (b, np.clip(b + blocksize, 0, h))])
    return blocks


def _complete_svd_from_blocks(block_U,block_SVT,blocks,k,dims, n_iter=15, random_state=42):

    # Compute the svd of the temporal components from all blocks
    from sklearn.utils.extmath import randomized_svd
    u, s, vt = randomized_svd(
        block_SVT.reshape([np.multiply(*block_SVT.shape[:2]),-1]),
        n_components=k,
        n_iter=n_iter,
        power_iteration_normalizer ='QR',
        random_state=random_state)

    S = s;
    SVT = np.dot(np.diag(S),vt)

    # Map the blockwise spatial components compontents to the second SVD
    U = np.dot(assemble_blockwise_spatial(block_U,blocks,dims),u)
    return U,SVT,S


def assemble_blockwise_spatial(block_U,blocks,dims):
    w,h = dims
    U = np.zeros([block_U.shape[0],block_U.shape[-1],w,h],dtype = 'float32')
    weights = np.zeros((w,h),dtype='float32')
    for iblock,(i,j) in enumerate(blocks):
        lw,lh = (i[1]-i[0],j[1]-j[0])
        U[iblock,:,i[0]:i[1],j[0]:j[1]] = block_U[iblock,:lw,:lh,:].transpose(-1,0,1)
        weights[i[0]:i[1],j[0]:j[1]] += 1
    U = (U/weights).reshape((np.multiply(*U.shape[:2]),-1))
    return U.T


def svd_blockwise(dat, frames_average,
                  k=200, block_k=20,
                  blocksize=120, overlap=8,
                  random_state=42):

    '''
    Computes the blockwise single value decomposition for a matrix that does not fit in memory.
    U,SVT,S,(block_U,block_SVT,blocks) = svd_blockwise(dat,
                                                   frames_average,
                                                   k = 200,
                                                   block_k = 20,
                                                   blocksize=120,
                                                   overlap=8)
    dat is a [nframes X nchannels X width X height] array
    frames_average is a [nchannels X width X height] array; the average to be subtracted before computing the SVD
    k is the number of components to be extracted (randomized SVD)
    The blockwise implementation works by first running the SVD on overlapping chunks of the movie. Secondly,  SVD is ran on the extracted temporal components and the spatial components are scaled to match the actual frame size.
    The chunks have all samples in time but only a fraction of pixels.
    This is adapted from matlab code by Simon Musall.
    A similar approach is described in Stringer et al. Science 2019.
    Joao Couto - March 2020
    '''

    nframes, nchannels, w, h = dat.shape
    n = nframes * nchannels

    # Create the chunks where the SVD is ran initially,
    # these have all samples in time but only a few in space
    # chunks contain pixels that are nearby in space
    blocks = make_overlapping_blocks((w, h), blocksize=blocksize, overlap=overlap)
    nblocks = len(blocks)

    # M = U.S.VT
    # U are the spatial components in this case
    block_U = np.zeros((nblocks, blocksize, blocksize, block_k), dtype=np.float32)
    block_U[:] = np.nan

    # V are the temporal components
    block_SVT = np.zeros((nblocks, block_k, n), dtype=np.float32)
    block_U[:] = np.nan

    # randomized svd is ran on each chunk
    for iblock, (i, j) in tqdm(enumerate(blocks), total=len(blocks), desc='Computing SVD on data chunks:'):
        # subtract the average (this should be made the baseline instead)
        arr = np.array(dat[:, :, i[0]:i[1], j[0]:j[1]], dtype='float32')
        arr -= frames_average[:, i[0]:i[1], j[0]:j[1]]
        arr /= frames_average[:, i[0]:i[1], j[0]:j[1]]
        bw, bh = arr.shape[-2:]
        arr = arr.reshape([-1, np.multiply(*arr.shape[-2:])])
        u, s, vt = randomized_svd(arr.T,
                                  n_components=block_k,
                                  n_iter=5,
                                  power_iteration_normalizer='LQ',
                                  random_state=random_state)
        block_U[iblock, :bw, :bh, :] = u.reshape([bw, bh, -1])
        block_SVT[iblock] = np.dot(np.diag(s), vt)

    U, SVT, S = _complete_svd_from_blocks(block_U, block_SVT, blocks, k, (w, h))
    return U, SVT, S, (block_U, block_SVT, blocks)



def approximate_svd(dat, frames_average,
                    onsets=None,
                    k=500,
                    nframes_per_bin=30,
                    nbinned_frames=5000,
                    nframes_per_chunk=5000):

    '''
    Approximate single value decomposition by estimating U from the average movie and using it to compute S.VT.
    This is similar to what described in Steinmetz et al. 2017
    Joao Couto - March 2020
    TODO: Separate the movie binning from the actual SVD?
    '''

    if hasattr(dat, 'filename'):
        dat_path = dat.filename
    else:
        dat_path = None
    dims = dat.shape[1:]

    # the number of bins needs to be larger than k because of the number of components.
    if nbinned_frames < k:
        nframes_per_bin = np.clip(int(np.floor(len(dat) / (k))), 1, nframes_per_bin)
    nbinned_frames = np.min([nbinned_frames, int(np.floor(len(dat) / nframes_per_bin))])


    idx = np.arange(0, nbinned_frames * nframes_per_bin, nframes_per_bin, dtype='int')

    if not idx[-1] == len(dat):
        idx = np.hstack([idx, len(dat) - 1])
    binned = np.zeros([len(idx) - 1, *dat.shape[1:]], dtype='float32')
    for i in tqdm(range(len(idx) - 1), desc='Binning raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i + 1]]  # work when data are loaded to memory
        else:
            blk = load_binary_block((dat_path, idx[i], nframes_per_bin), shape=dims)
        avg = get_trial_baseline(idx[i], frames_average, onsets)
        binned[i] = np.mean((blk - avg + np.float32(1e-5)) / (avg + np.float32(1e-5)), axis=0)
    binned = binned.reshape((-1, np.multiply(*dims[-2:])))

    np.save("/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging/binned.npy", binned)
    binned = np.load("/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging/binned.npy")

    # Get U from the single value decomposition
    cov = np.dot(binned, binned.T) / binned.shape[1]
    cov = cov.astype('float32')

    u, s, v = svd(cov)
    U = normalize(np.dot(u[:, :k].T, binned), norm='l2', axis=1)
    k = U.shape[0]  # in case the k was smaller (low var)

    # if trials are defined, then use them to chunck data so that the baseline is correct
    if onsets is None:
        idx = np.arange(0, len(dat), nframes_per_chunk, dtype='int')
    else:
        idx = onsets
    if not idx[-1] == len(dat):
        idx = np.hstack([idx, len(dat) - 1])
    V = np.zeros((k, *dat.shape[:2]), dtype='float32')


    # Compute SVT
    for i in tqdm(range(len(idx) - 1), desc='Computing SVT from the raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i + 1]]  # work when data are loaded to memory
        else:
            blk = load_binary_block((dat_path, idx[i], idx[i + 1] - idx[i]), shape=dims).astype('float32')
        avg = get_trial_baseline(idx[i], frames_average, onsets).astype('float32')
        blk = (blk - avg + np.float32(1e-5)) / (avg + np.float32(1e-5))
        V[:, idx[i]:idx[i + 1], :] = np.dot(
            U, blk.reshape([-1, np.multiply(*dims[1:])]).T).reshape((k, -1, dat.shape[1]))

    SVT = V.reshape((k, -1))
    U = U.T.reshape([*dims[-2:], -1])


    return U, SVT


base_directory = "/media/matthew/29D46574463D2856/NXAK14.1A_2021_06_15_Transition_Imaging"
blue_file = "NXAK14.1A_20210615-135401_Blue_Data.hdf5"
violet_file = "NXAK14.1A_20210615-135401_Violet_Data.hdf5"

blue_mean = np.load(os.path.join(base_directory, "blue_mean.npy"))
blue_mean = np.reshape(blue_mean, (600, 608))
#plt.title("Blue Mean")
#plt.imshow(blue_mean)
#plt.show()

violet_mean = np.load(os.path.join(base_directory, "violet_mean.npy"))
violet_mean = np.reshape(violet_mean, (600, 608))
#plt.title("Violet Mean")
#plt.imshow(violet_mean)
#plt.show()

blue_mean = np.swapaxes(blue_mean, 0, 1)
violet_mean = np.swapaxes(violet_mean, 0, 1)
frame_average = np.array([blue_mean, violet_mean])

"""
interleaved_file_object = h5py.File(os.path.join(base_directory, "Interleaved_Array.hdf5"), 'r')
data_matrix = interleaved_file_object["Data"]
"""

"""
interleaved_file_object = tables.open_file(os.path.join(base_directory, "Interleaved_Tables_Motion_Correction.h5"), mode='r')
data_matrix = interleaved_file_object.root.Data

data_matrix_sample = data_matrix[0:10000]
print("Data Matrix Shape", np.shape(data_matrix))

U, SVT, = approximate_svd(data_matrix_sample, frame_average)
np.save(os.path.join(base_directory, "SVT_Churchland_Aprox.npy"), SVT)
np.save(os.path.join(base_directory, "U_Churchland_Aprox.npy"), U)
"""
svt = np.load(os.path.join(base_directory, "SVT_Churchland_Aprox.npy"))
u = np.load(os.path.join(base_directory, "U_Churchland_Aprox.npy"))
print("SVT", np.shape(svt))
print("U", np.shape(u))

reconstructed_data = np.dot(u, svt[:, 0:2000])
print("Reconstructed Data", np.shape(reconstructed_data))
print("Min", np.min(reconstructed_data))
print("Max", np.max(reconstructed_data))

vmin = np.percentile(reconstructed_data, q=5)
vmax = np.percentile(reconstructed_data, q=99)

plt.ion()
figure_1 = plt.figure()
for frame in range(0, 2000, 2):

    blue_frame_data = reconstructed_data[:, :, frame]
    violet_frame_data = reconstructed_data[:, :, frame + 1]

    violet_frame_data = np.transpose(violet_frame_data)
    blue_frame_data = np.transpose(blue_frame_data)
    difference_frame = np.subtract(blue_frame_data, violet_frame_data)

    rows = 1
    columns = 3

    blue_axis = figure_1.add_subplot(rows, columns, 1)
    violet_axis = figure_1.add_subplot(rows, columns, 2)
    difference_axis = figure_1.add_subplot(rows, columns, 3)

    blue_axis.imshow(blue_frame_data, vmin=0, vmax=vmax)
    violet_axis.imshow(violet_frame_data, vmin=0, vmax=vmax)
    difference_axis.imshow(difference_frame, vmin=-0.1, vmax=0.1)

    plt.draw()
    plt.pause(0.1)
    plt.clf()