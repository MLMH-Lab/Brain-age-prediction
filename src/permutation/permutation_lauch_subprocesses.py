#!/usr/bin/env python3
import argparse
import multiprocessing as mp
from pathlib import Path

from permutation_train_svm_models import train

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='homogenized_ids.csv',
                    help='Filename indicating the ids to be used.')

args = parser.parse_args()


def main(experiment_name, scanner_name, input_ids_file):
    n_threads = 2
    initial_index_perm = 0
    number_perm_per_thread = 40

    processes = []
    for i in range(n_threads):
        processes.append(mp.Process(target=train, args=(experiment_name, scanner_name,
                                     input_ids_file,
                                     (initial_index_perm + i * number_perm_per_thread),
                                     (initial_index_perm + (i + 1) * number_perm_per_thread))))

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()



if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
