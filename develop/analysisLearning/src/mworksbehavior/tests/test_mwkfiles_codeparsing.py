"""This module tests parsing of raw code streams, before h5 files are generated

We extract the raw code streams from the mwk/2 files to save space in this repository

"""

import pytest  # this is the future; transition tests to this
import numpy as np
import sys

import joblib # dependency for reading pickles
assert (sys.version_info[0] == 3), 'Joblib pickles not compatible with Python v2' # created on python 3, see joblib docs

from mworksbehavior import mwkfiles

@pytest.mark.filterwarnings('ignore:MWTrialTimestamp zero')
def test_digital_stream1():
    ds = joblib.load('data/180713-test-digcodes.pkl.gz')
    trCodeL, mwTrialTsUs, discardedD = mwkfiles.parse_digital_stream(ds['codenums'], ds['ts'], tryToFixFile=True)

    # quick checks.  For the most part we're testing if above code throws errors on this stream
    assert len(trCodeL) == 27
    assert len(trCodeL[26]) == 21
    # below line tests that we get a series and then uses the .values ndarray for comparison
    np.testing.assert_equal(trCodeL[26].iloc[0:5].values, [170, 170, 170, 200, 0])  # ints so we don't need allclose



def test_serial_stream1():
    """as above most testing is done by the function actually returning without error"""
    ds = joblib.load('data/180720-test-sercodes.pkl.gz')

    (mwEndStateDf, mwEventDf, codecS) = mwkfiles.parse_blackrock_serial_stream(ds['serCodes'], ds['serTs'])

    assert len(mwEndStateDf) == 27
    np.testing.assert_equal(mwEndStateDf.tTotalReqHoldTimeMs[0:3].values.astype('int'), [1000,2127,3150])
    desIx = (mwEventDf.codename == 'strobedDigitalWord') & (mwEventDf.value == 170) & (mwEventDf.trialNum == 2)
    np.testing.assert_equal(mwEventDf.mwTimestampUs[desIx].iloc[0:3].values, [51637114,51637895,51638414])
    assert codecS[24] == 'debuggerStep'

