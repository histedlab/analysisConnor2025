#!/usr/bin/env python

# pandas 1.0.5 does not read old HDF5 files
# https://github.com/pandas-dev/pandas/issues/33186

# This script should be run in a WORKING environment:
# reads the h5 file and writes the equiv h5

import argparse
from pathlib import Path
import pandas as pd


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file', help='input h5 file to be converted')
    args = parser.parse_args()

    outPklName = Path(args.h5file).with_suffix('.pkl.gz')

    df = pd.read_hdf(args.h5file)
    df.to_pickle(outPklName, compression='gzip')

    print(f'Done, wrote {outPklName}')

    
if __name__ == '__main__':
    convert()

