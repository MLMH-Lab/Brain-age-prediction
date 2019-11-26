"""
Script to create the Kernel matrix (Gram matrix) that will be used on the analysis with
voxel data.
"""
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
import pandas as pd

PROJECT_ROOT = Path.cwd()

# def main():
# ------------------------------------------------------------------------------
# CHANGE HERE
# ------------------------------------------------------------------------------
dataset_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1'
kernel_path = PROJECT_ROOT / 'outputs' / 'kernels'
input_data_type = '.nii.gz'

# ------------------------------------------------------------------------------
# Create output folder if it does not exist
kernel_path.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------------
# Get the same subject's IDs for those used on the FreeSurfer analysis and make
# sure that we have IDs for which we have age and images
# ------------------------------------------------------------------------------
freesurfer_ids_file = 'cleaned_ids.csv'
freesurfer_ids_path = PROJECT_ROOT / 'outputs' / 'biobank_scanner1' / freesurfer_ids_file

try:
    subject_ids = pd.read_csv(freesurfer_ids_path)
except IOError:
    print('No file {}. Run the clean_biobank1_data.py script to generate it.'.format(freesurfer_ids_file))
    raise

# Get list of subjects for which we have data
subjects_path = [str(dataset_path / '{}_ses-bl_T1w_Warped{}'.format(subject_id, input_data_type)) for subject_id in subject_ids['Participant_ID']]

n_samples = len(subject_ids)
print('Total number of images: {}'.format(len(subject_ids)))

# Calculate the Gram matrix
# ------------------------------------------------------------------------------
# Load the mask image
brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
mask_img = nib.load(str(brain_mask))

K = np.float64(np.zeros((n_samples, n_samples)))
step_size = 30
images = []

# Outer loop
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
        try:
            img = nib.load(str(path))
        except FileNotFoundError:
            print('No image file {}.'.format(path))
            raise

        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float64')
        img = np.nan_to_num(img)
        images_1.append(img)
        del img
    images_1 = np.array(images_1)

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
                try:
                    img = nib.load(str(path))
                except FileNotFoundError:
                    print('No image file {}.'.format(path))
                    raise

                img = apply_mask(img, mask_img)
                img = np.asarray(img, dtype='float64')
                img = np.nan_to_num(img)
                images_2.append(img)
                del img
            images_2 = np.array(images_2)

        block_K = np.dot(images_1, np.transpose(images_2))
        K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
        K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

gram_df = pd.DataFrame(columns=subject_ids, data=K)
gram_df['Participant_ID'] = subject_ids
gram_df = gram_df.set_index('Participant_ID')

print("")
print("Saving kernel for this dataset")
print("   Kernel Path: {}".format(str(kernel_path)))
gram_df.to_csv((kernel_path / 'kernel.csv'))
print("Done")
#
#
# if __name__ == "__main__":
#     main()
