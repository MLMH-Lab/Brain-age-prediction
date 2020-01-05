""" Script to save images of the importance of the voxels. """
from pathlib import Path
from glob import glob

import imageio
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import coord_transform

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

    # Load brain mask and find index for the voxels inside the mask
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
    mask_img = nib.load(str(brain_mask))
    mask_data = mask_img.get_fdata()
    coords = np.argwhere(mask_data > 0)

    importance_map = np.zeros_like(mask_data)
    for xyz, importance in zip(coords, final_coefs):
        importance_map[tuple(xyz)] = importance

    # --------------------------------------------------------------------------
    importance_map_nifti = nib.Nifti1Image(importance_map, mask_img.affine)

    # Select central slice for the X and Y coordinate
    x = importance_map_nifti.shape[0]/2
    y = importance_map_nifti.shape[1]/2

    # Define min and max slice to analyse
    z_min = 28
    z_max = 160

    # Load anatomical template image
    template = (PROJECT_ROOT / 'imaging_preprocessing_ANTs' /
    'mni_icbm152_t1_tal_nlin_sym_09c_brain.nii.gz')
    template = nib.load(str(template))

    # Save multiple slices
    for i in range(z_min, z_max, 1):
        # transform slice into into the image space
        coordinates = coord_transform(x, y, i, importance_map_nifti.affine)
        print(coordinates)
        plotting.plot_stat_map(importance_map_nifti,
                               bg_img=template,
                               cut_coords=coordinates,
                               output_file=feat_imp_dir /
                               'feature_importance_{}.png'.format(i),
                               draw_cross=False,
                               cmap='Reds',
                               threshold=thr_mean + 2 * thr_std,
                               black_bg=False,
                               )
    # Make gif
    png_list = glob(str(feat_imp_dir / 'feature_importance*.png'))
    png_list.sort()
    gif_file = feat_imp_dir / 'feature_importance.gif'
    with imageio.get_writer(gif_file, mode='I') as writer:
        for fname in png_list:
            image = imageio.imread(fname)
            writer.append_data(image)

if __name__ == "__main__":
    main()
