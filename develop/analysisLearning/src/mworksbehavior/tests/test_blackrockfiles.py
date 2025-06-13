import pytest  # this is the future; transition tests to this
import numpy as np
import sys
import os
import tempfile
import warnings

import gzip
#import joblib # dependency for reading pickles
#assert (sys.version_info[0] == 3), 'Joblib pickles not compatible with Python v2' # created on python 3, see joblib docs

warnings.filterwarnings("ignore", "Using or importing the ABCs from 'collections'") # DeprecationWarning

from .. import mwkfiles
from .. import blackrockfiles


def decompress_datafile(compname):
    """decompress the nev file - in future should extend to allow the BlackrockFile classes to read compressed
    files directly
    store it compressed to save space in repo
    save output in a temp dir
    Returns:
        decompressed temporary filename
    """
    uncompname = os.path.join(tempfile.gettempdir(),
                              os.path.basename(compname).rsplit('.gz', 1)[0])  # rsplit() does replace() at end only
    with gzip.GzipFile(compname) as gfh:
        s = gfh.read()
    with open(uncompname, 'wb') as outfh:
        outfh.write(s)
    return uncompname

@pytest.mark.filterwarnings('ignore:no nsX')
@pytest.mark.filterwarnings('ignore:MWTrialTimestamp zero')
@pytest.mark.filterwarnings('ignore:Method .ptp is deprecated') # thrown by np.isclose() I think
def test_read1():
    uncompname = decompress_datafile('data/180713-datafile002.nev.gz')

    bfs = blackrockfiles.BlackrockFileWithSerial(uncompname, tryToFixFile=True)

    # should do some more verification of parsing here... todo
    assert(bfs.nTrials == 27)
    exEvent = bfs.get_mw_event(trialNum=1, codename='laserTriggerFIO', value=1)
    assert (np.isclose(exEvent.brTimestampS,28.978987))


@pytest.mark.filterwarnings('ignore:no nsX')
@pytest.mark.filterwarnings('ignore:MWTrialTimestamp zero')
@pytest.mark.filterwarnings('ignore:Method .ptp is deprecated') # thrown by np.isclose() I think
def test_read_pulselasertest_180806():
    uncompname = decompress_datafile('data/180806-datafile001.nev.gz')

    bfs = blackrockfiles.BlackrockFileWithSerial(uncompname, tryToFixFile=True)

    # should do some more verification of parsing here... todo
    print(bfs.nTrials)
    assert (bfs.nTrials == 21)  # after truncation - in this file one extra MWorks trial
    exEvent = bfs.get_mw_event(trialNum=1, codename='laserTriggerFIO', value=1)
    print(exEvent.brTimestampS)
