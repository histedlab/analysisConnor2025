import numpy as np
import pytest

r_ = np.r_
import os
import sys
import subprocess as sp

from mworksbehavior import mwk_io
from mworksbehavior import mwkfiles


rootdir = "data"


# as of MWorks 0.10 (Feb 2020), MWorks now supports versions >= 3.5
#def test_python_ver():
#    assert (
#        sys.version_info[0] >= 3 and sys.version_info[1] == 7
#    ), "MWorks Framework depends on Py 3.7"


# Future: can parametrize this fixture to read several data files in sequence
@pytest.fixture
def mwk2_file():
    """uncompress mwk file, return mwk2 file name in tmp"""
    mname = "190524_i2086_stim0.mwk2"
    fullname = os.path.join(rootdir, mname + ".gz")
    sp.check_call("cp %s /tmp/" % fullname, shell=True)
    sp.check_call("gunzip --force %s" % os.path.join("/tmp", mname + ".gz"), shell=True)
    return os.path.join("/tmp", mname)

def test_mwk_to_df(mwk2_file):

    mwk_io.mwk_to_h5(
        mwk_file=mwk2_file, out_file=None, keep_system_vars=True, exist_delete=True
    )
    tn = "/tmp/outfile.h5"
    if os.path.exists(tn):
        os.unlink(tn)

    mwk_io.mwk_to_h5(
        mwk_file=mwk2_file, out_file=tn, keep_system_vars=True, exist_delete=False
    )

    # some simple checks
    fulln = "/tmp/outfile.h5"
    mwf = mwkfiles.RetinotopyMap1MWKFile(
        fulln, use_counter_as_start=True
    )  # defaults
    mwf.compute_imaging_constants()

    # below breaks on 250120 changes due to dropping zero code, just use the startendtimes version which is right
    #df0 = mwf.get_codedf('counterFIO', only_in_startend_times=False)
    #assert (len(df0) == 817)
    df0 = mwf.get_codedf('counterFIO', only_in_startend_times=True)
    print(mwf.counterStats)
    assert (len(df0) == 810 and mwf.counterStats.nCounterTicks == 810)

    assert len(mwf.stimDf) == 27
