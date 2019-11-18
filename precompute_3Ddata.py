import glob
from pathlib import Path
import re

import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------------
# CHANGE HERE
# ---------------------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
# TODO: Change the path. Here you should load the voxel data?
dataset_path = Path('/Volumes/Elements/BIOBANK/SCANNER01')
# sites_path = "./data/sites3.csv"
kernel_file = './outputs/kernels/img.npz'
input_data_type = "*.nii.gz"

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

demographics = pd.read_csv((dataset_path / 'participants.tsv'), sep='\t')
demographics.set_index('Participant_ID', inplace=True)
print("Reading images with format {} from: %s".format(input_data_type,
                                                      dataset_path))

# Get list of subjects for which we have data
subjects_path = glob.glob(str(dataset_path / 'sub-*Warped.nii.gz'))
subjects_path.sort()
# Get only the subject's ID from their path
subjects_id = [re.findall('sub-\d+', subject)[0] for subject in
               subjects_path]
# TODO: Get a list of 100 subjects, for testing purposes
subjects_id = subjects_id[:100]
subjects_path = subjects_path[:100]

# Get demographics only for the subjects we have information for
demographics = demographics[demographics.index.isin(subjects_id)]

# n_samples = len(labels)
# if n_samples != len(paths_train):
#     raise ValueError('Different number of labels and images files')
# Select demographics for group of interst

print("Loading images")
print("   # of images samples: %d " % len(subjects_id))

n_samples = len(subjects_id)

# Load the mask image
brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / \
             'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
mask_img = nib.load(str(brain_mask))

K = np.float64(np.zeros((n_samples, n_samples)))
step_size = 30
images = []

# outer loop
for ii in range(int(np.ceil(n_samples / np.float(step_size)))):

    it = ii + 1
    max_it = int(np.ceil(n_samples / np.float(step_size)))
    print(" outer loop iteration: %d of %d." % (it, max_it))

    # generate indices and then paths for this block
    start_ind_1 = ii * step_size
    stop_ind_1 = min(start_ind_1 + step_size, n_samples)
    block_paths_1 = subjects_path[start_ind_1:stop_ind_1]

    # read in the images in this block
    images_1 = []
    for k, path in enumerate(block_paths_1):
        img = nib.load(path)
        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float64')
        img = np.nan_to_num(img)
        images_1.append(img)
        del img
    images_1 = np.array(images_1)
    # Normalise the intensity for each subject
    # images_1 = MinMaxScaler(images_1, axis=1)
    for jj in range(ii + 1):

        it = jj + 1
        max_it = ii + 1

        print(" inner loop iteration: %d of %d." % (it, max_it))

        # if ii = jj, then sets of image data are the same - no need to load
        if ii == jj:

            start_ind_2 = start_ind_1
            stop_ind_2 = stop_ind_1
            images_2 = images_1

        # if ii !=jj, read in a different block of images
        else:
            start_ind_2 = jj * step_size
            stop_ind_2 = min(start_ind_2 + step_size, n_samples)
            block_paths_2 = subjects_path[start_ind_2:stop_ind_2]

            images_2 = []
            for k, path in enumerate(block_paths_2):
                img = nib.load(path)
                img = apply_mask(img, mask_img)
                img = np.asarray(img, dtype='float64')
                img = np.nan_to_num(img)
                images_2.append(img)
                del img
            images_2 = np.array(images_2)
            # images_2 = normalize(images_2, axis=1)

        block_K = np.dot(images_1, np.transpose(images_2))
        K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
        K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)
        # plt.imshow(K)
        # plt.savefig('k-matrix.png')
print("")
print("Saving Dataset")
print("   Kernel+Labels:" + kernel_file)
np.savez(kernel_file, kernel=K)
print("Done")
