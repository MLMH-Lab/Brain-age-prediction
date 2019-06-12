""""""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    n_bootstrap = 1000
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    scores_i_n_subjects_mean = []
    scores_i_n_subjects_std = []
    for i_n_subjects in range(1, 51):
        ids_with_n_subjects_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subjects))
        scores_dir = ids_with_n_subjects_dir / 'scores'
        scores_bootstrap = []
        for i_bootstrap in range(n_bootstrap):
            # Save arrays with permutation coefs and scores as np files
            filepath_scores = scores_dir / ('boot_scores_{:04d}.npy'.format(i_bootstrap))
            scores_bootstrap.append(np.load(str(filepath_scores))[1])

        scores_i_n_subjects_mean.append(np.mean(scores_bootstrap))
        scores_i_n_subjects_std.append(np.std(scores_bootstrap))

    train_sizes = range(1,51)

    age_min = 1
    age_max = 40
    std = np.sqrt(((age_max-age_min)**2)/12)

    # Draw lines
    plt.plot(train_sizes, scores_i_n_subjects_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, std*np.ones_like(train_sizes), color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, scores_i_n_subjects_mean - 1.96*scores_i_n_subjects_std, scores_i_n_subjects_mean + 1.96*scores_i_n_subjects_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
