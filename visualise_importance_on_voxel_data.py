""" Script to save images of the importance of the voxels. """
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # --------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'voxel_SVM'
    svm_dir.mkdir(exist_ok=True, parents=True)
    cv_dir = svm_dir / 'cv'
    # --------------------------------------------------------------------------

    assessed_model_coefs = []
    n_repetitions = 1
    for i_repetition in range(n_repetitions):
        weights_filename = '{:02d}_weights.npy'.format(i_repetition)
        coefs = np.load(cv_dir / weights_filename)
        assessed_model_coefs.append(coefs)

    assessed_model_coefs = np.array(assessed_model_coefs)
    assessed_model_coefs = assessed_model_coefs.reshape(-1, assessed_model_coefs.shape[-1])

    assessed_mean_relative_coefs = np.divide(np.abs(assessed_model_coefs),
                                             np.sum(np.abs(assessed_model_coefs), axis=1)[:, np.newaxis])
    final_coefs = np.mean(assessed_mean_relative_coefs, axis=0)
    # --------------------------------------------------------------------------

    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
    mask_img = nib.load(str(brain_mask))
    mask_data = mask_img.get_fdata()
    coords = np.argwhere(mask_data > 0)

    importance_map = np.zeros_like(mask_data)
    for xyz, importance in zip(coords, final_coefs):
        importance_map[tuple(xyz)] = importance

    # --------------------------------------------------------------------------
    importance_map_nifti = nib.Nifti1Image(importance_map, np.eye(4))
    #
    # plotting.plot_glass_brain(importance_map_nifti, threshold=0)
    # plt.show()


if __name__ == "__main__":
    main()
