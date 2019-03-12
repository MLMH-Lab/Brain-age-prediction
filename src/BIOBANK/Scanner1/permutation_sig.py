"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np
import pandas as pd

import os
import glob

from sklearn.externals import joblib

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Define what subjects were modeled: total, male or female
    subjects = 'total'

    # Load permutation coefficients
    perm_coef = np.load(PROJECT_ROOT / 'outputs' / 'permutations' / subjects / 'perm_coef.npy')

    # Get number of permutations run
    n_perm = len(perm_coef)

    # Get file names of all SVM models run
    model_file_names = []
    output_dir = str(PROJECT_ROOT / 'outputs' / subjects)
    for file_path in glob.iglob(output_dir + '*/*svm.joblib', recursive=True):
        file_name = os.path.basename(file_path)
        model_file_names.append(file_name)

    # Get number of all SVM models run, specify number of features per model
    n_models = len(model_file_names)
    n_features = 100

    # Get coefficients of all SVM models run in an array of dimensions n_models * n_features
    actual_model_coef = np.zeros([n_models, n_features])
    index = 0
    for model_name in model_file_names:
        model = joblib.load(PROJECT_ROOT / 'outputs' / subjects / model_name)
        model_coef = model.coef_
        actual_model_coef[index] = model_coef
        index += 1

    # Get mean of absolute SVM model coefficients
    actual_model_coef_abs = np.abs(actual_model_coef)
    actual_model_coef_mean = np.mean(actual_model_coef_abs, axis=0)

    # Calculate p-value per feature:
    # number of times the perm coef is >= model coef divided by number of permutations

    # Create array of permuted coef subtracted by mean of actual coef
    diff_array = np.zeros([n_perm, n_features])

    # Populate diff_array
    ind = 0
    for row in perm_coef:
        new_row = row - actual_model_coef_mean
        diff_array[ind] = new_row
        ind += 1

    # Calculate p-value per feature
    df = pd.DataFrame(diff_array)
    list = []

    for label, row in df.iteritems():
        filtered_df = df[df[label] <= 0]
        n_coef_higher = len(filtered_df)
        p_val = n_coef_higher / n_perm
        list_item = [label, n_coef_higher, p_val]
        list.append(list_item)

    p_df = pd.DataFrame(list, columns=['Feature', 'N_permcoef_higher', 'p_val'])



if __name__ == "__main__":
    main()