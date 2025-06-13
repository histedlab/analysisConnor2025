import numpy as np
import pytest

r_ = np.r_
import warnings
import os
#import sys

# warnings thrown via imports
warnings.filterwarnings("ignore", "Using or importing the ABCs from 'collections'") # DeprecationWarning
warnings.filterwarnings("ignore", "zmq.eventloop.ioloop is deprecated in pyzmq 17") # DeprecationWarning

from mworksbehavior import mwkfiles
from mworksbehavior.imaging import intrinsic as ii

rootdir = "data"

retinodatafiles = ['190529_i1951_stim1', # RM2 file
                   '190502_i2026_stim0', # RM1 file
                    ]

@pytest.fixture
def fake_stack():
    """generate a fake image stack on the fly so we don't need to keep it on disk"""
    desShape = (9,30,128,128)
    fstack = np.zeros(desShape, dtype='uint16') + 1 # avoid divide by zero
    fstack[1,15:30,100:110,100:110] = 30
    return fstack

@pytest.fixture(params=retinodatafiles)
def mwf_read(request):
    h5name = 'data/%s.h5' % request.param
    mwf = mwkfiles.RetinotopyMap1MWKFile(h5name, stims_to_keep='all')
    mwf.save_stim_params('/tmp/stimout.h5')  # just make sure writing code doesn't throw
    mwf.compute_imaging_constants()
    return mwf

@pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
@pytest.mark.filterwarnings("ignore:elementwise comparison failed:FutureWarning") # mpl
def test_mwk_to_df(fake_stack, mwf_read):
    mwf = mwf_read
    allstimavg = fake_stack

    c = mwf.constS
    dfofmax = 100
    pre_chop = 5
    base_range = r_[pre_chop:c.counterNPre]
    stim_range = r_[c.counterNPre:c.counterNPre + c.counterNStim]
    #print('base {}-{}\nstim {}-{}'.format(base_range.min(), base_range.max(), stim_range.min(), stim_range.max()))
    fig = ii.plot_stim_maps(allstimavg, mwf, dfofRange=r_[-1, 1] * dfofmax, base_range=base_range,
                            stim_range=stim_range)

    os.makedirs('figout', exist_ok=True)
    fig.savefig('figout/intrinsic_plot_stim_maps-%s.pdf'%mwf.basename)

