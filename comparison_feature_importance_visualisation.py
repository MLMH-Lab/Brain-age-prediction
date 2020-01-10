#!/usr/bin/env python3
""" Script to save images of the importance of the voxels. """
from pathlib import Path
from glob import glob

import imageio
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import coord_transform
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # --------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'voxel_SVM'
    svm_dir.mkdir(exist_ok=True, parents=True)
    # Path where feature importance will be saved
    feat_imp_dir = svm_dir / 'feature_importance'
    feat_imp_dir.mkdir(exist_ok=True, parents=True)
    cv_dir = svm_dir / 'cv'
    print(feat_imp_dir)

    # Load anatomical template image
    template = (PROJECT_ROOT / 'imaging_preprocessing_ANTs' /
                'mni_icbm152_t1_tal_nlin_sym_09c.nii')
    template = nib.load(str(template))

    brain_mask = (PROJECT_ROOT / 'imaging_preprocessing_ANTs' /
                  'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii')
    mask_img = nib.load(str(brain_mask))
    bg_img = template.get_fdata() * mask_img.get_fdata()
    template = nib.Nifti1Image(bg_img, template.affine)

    # --------------------------------------------------------------------------
    assessed_model_coefs = []
    n_repetitions = 10
    n_folds = 10
    for i_repetition in range(n_repetitions):
        print('Repetition N={}'.format(i_repetition))
        for i_fold in tqdm(range(n_folds)):
            importance_filename = '{:02d}_{:02d}_importance.nii.gz'.format(i_repetition, i_fold)
            importance_map = nib.load(str(cv_dir / importance_filename))
            data_map = importance_map.get_fdata()
            assessed_model_coefs.append(np.abs(data_map) / np.sum(np.abs(data_map), axis=(0, 1, 2)))

    assessed_model_coefs = np.array(assessed_model_coefs)
    final_coefs = np.mean(assessed_model_coefs, axis=0)

    # --------------------------------------------------------------------------
    # Create mean importance image
    importance_map_nifti = nib.Nifti1Image(final_coefs, template.affine)

    # Select central slice for the X and Y coordinate
    x = importance_map_nifti.shape[0] / 2
    y = importance_map_nifti.shape[1] / 2

    # Define min and max slice to analyse
    z_min = 28
    z_max = 160

    # Define threshold for visualisation purposes
    thr_mean = np.mean(final_coefs)
    thr_std = np.std(final_coefs)

    # Save multiple slices
    print('Saving images with different z-coordinates:')
    for i in tqdm(range(z_min, z_max, 1)):
        # transform slice into into the image space
        coordinates = coord_transform(x, y, i, importance_map_nifti.affine)
        plotting.plot_stat_map(importance_map_nifti,
                               bg_img=template,
                               cut_coords=coordinates,
                               output_file=feat_imp_dir /
                                           'feature_importance_{}.png'.format(i),
                               draw_cross=False,
                               cmap='Reds',
                               threshold=thr_mean + 3 * thr_std,
                               black_bg=False,
                               )
    # Make gif
    print('Create GIF')
    png_list = glob(str(feat_imp_dir / 'feature_importance*.png'))
    png_list.sort()
    gif_file = feat_imp_dir / 'feature_importance.gif'
    with imageio.get_writer(gif_file, mode='I') as writer:
        for fname in png_list:
            image = imageio.imread(fname)
            writer.append_data(image)


if __name__ == "__main__":
    main()