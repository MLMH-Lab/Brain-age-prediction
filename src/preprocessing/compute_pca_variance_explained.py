"""Script to calculate the % variance explained from the PCA models"""

import pandas as pd
from joblib import load
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    pca_path = PROJECT_ROOT / 'outputs' / 'pca' / 'models'

    pca_name_ls = []
    for i_repetition in range(10):
        for i_fold in range(10):
            pca_name = f'{i_repetition:02d}_{i_fold:02d}_pca.joblib'
            pca_name_ls.append(pca_name)

    pca_dict = {}
    for i_model in pca_name_ls:
        pca_model = load(pca_path / i_model)
        var_explained = pca_model.explained_variance_ratio.sum()
        pca_dict[pca_model] = var_explained

    pca_df = pd.DataFrame.from_dict(pca_dict)
    file_name = 'pca_variance_explained.csv'
    file_path = PROJECT_ROOT / 'outputs' / 'pca' / file_name
    pca_df.to_csv(file_path / file_name)



if __name__ == '__main__':
    main()