#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

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

parser.add_argument('-T', '--script_filename',
                    dest='script_filename',
                    default='permutation_train_svm_models.py',
                    help='Filename indicating the script to be used.')

args = parser.parse_args()


def main(experiment_name, scanner_name, script_filename):
    venv_python = PROJECT_ROOT / 'venv' / 'bin' / 'python3'
    code_perm = PROJECT_ROOT / script_filename

    n_threads = 5
    initial_index_perm = 0
    number_perm_per_thread = 200

    slices_indexes = []
    for i in range(n_threads):
        slices_indexes.append(
            [initial_index_perm + i * number_perm_per_thread, initial_index_perm + (i + 1) * number_perm_per_thread])

    processes = []

    for slices_i in slices_indexes:
        command = (f'{str(venv_python)} {str(code_perm)} -E {experiment_name}'
                   f' -S {scanner_name} -K {slices_i[0]:d} -L {slices_i[1]:d}')
        print(command)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.input_ids_file)
