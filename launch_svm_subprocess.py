from pathlib import Path
import subprocess

PROJECT_ROOT = Path.cwd()
venv_python = PROJECT_ROOT / 'venv' / 'bin' / 'python3'
code_perm = PROJECT_ROOT / 'train_svm_on_voxel_data.py'

n_threads = 10
initial_index_perm = 0
number_perm_per_thread = 1

processes = []

for repetition in range(n_threads):
    command = str(venv_python) + ' ' + str(code_perm) + ' %d' % (repetition)
    print(command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
    processes.append(p)

for p in processes:
    p.wait()
