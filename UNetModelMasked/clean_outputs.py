import os
from glob import glob

folders = ['logs/*/', 'store_models', 'output_products', 'losses', '__pycache__']

for folder in folders:
    files = glob(os.path.join(folder, '*'))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

logfile = 'train.log'
if os.path.isfile(logfile):
    os.remove(logfile)
