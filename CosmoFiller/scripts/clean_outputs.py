import os
from glob import glob

main_dir = 'train_delta'
folders = ['logs/*/', 'store_models', 'output_data', 'losses', '__pycache__']

for folder in folders:
    files = glob(os.path.join(main_dir, folder, '*'))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
            print(f'Removed file: {f}')

logfile = os.path.join(main_dir, 'train.log')
if os.path.isfile(logfile):
    os.remove(logfile)
    print(f'Removed file: {logfile}')
print('Cleanup completed.')