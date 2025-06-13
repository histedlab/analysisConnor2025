import numpy as np
import pytest
import os
import sys
from pathlib import Path
import dill as pickle
from ..console_entry_points import parse_mwk_retino as pmr

r_ = np.r_




def test_onefile():
    nameP = Path('210725-i3689-stim2.mwk2')
    fileP = Path('data') / nameP
    outP = Path('/tmp')
    pmr.main([fileP.as_posix(), outP.as_posix()])  # argv parameter is a list

    # read it
    with open(outP / (nameP.stem + '-mwk2.pkl'), 'rb') as fp:
        dataD = pickle.load(fp)
        assert len(dataD) == 8
        assert dataD['nPreFr'] == 60