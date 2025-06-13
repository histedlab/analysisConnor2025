import scipy.io
import pytoolsMH
from pytoolsMH import containers
import pandas as pd
import numpy as np
import re
import warnings
import logging

log = logging.getLogger(__name__)


def closeto(x, y, eps=1e-10):
    return np.abs(np.array(x) - np.array(y)) < eps


class matBehavFile:
    """Read a MWorks mat data file into a dataframe
    
    attributes:
        df - the concatenated data frame (can have multiple blocks)
        miscDL - len(nBlocks) - misc fields (e.g. savedEvents) from each block
        constDL - len(nBlocks) - constants from each block
    """

    def __init__(self, fName):

        matD = scipy.io.loadmat(fName, squeeze_me=True)

        # parse date str from filename
        g0 = re.match(".*data2-i([0-9]*)-([0-9]*).mat$", fName).groups()
        self.subjectNum = int(g0[0])
        self.dateStr = g0[1]

        # get blocks from file
        blockL = [matD["input"]]
        if not "backup" in matD:
            pass  # no extra data
        elif containers.lenorzero(matD["backup"]) == 0:
            # single backup structure
            blockL = blockL + [matD["backup"]]
        else:
            assert 0, "unclear how to handle this backup type, work on it"

        # construct a df from each block, then append
        dfL = []
        self.miscDL = []
        self.constDL = []
        for (iB, tB) in enumerate(blockL):
            (tDf, tMiscD, tConstD) = self._oneBlockFileDataToDf(tB)
            dfL.append(tDf)
            self.miscDL.append(tMiscD)
            self.constDL.append(tConstD)
            log.info("num trials: %d" % len(dfL[-1]))
            if iB > 0:
                warnings.warn("We need to merge miscD and constD from multiple blocks - todo")

        self.df = pd.concat(dfL)

    def _oneBlockFileDataToDf(self, matStruct):
        in0 = matStruct

        dfD = {}
        constD = {}
        miscD = {}
        nTrials = len(in0["trialOutcomeCell"].item())
        for (iN, tN) in enumerate(in0.dtype.names):
            tI = in0[tN].item()
            if type(tI) == np.ndarray and containers.lenorzero(tI) == 0 and tI.dtype.fields is not None:
                # record array, but singleton, so unpack it
                tI = tI.item()
            else:
                # try converting to numeric if possible, using pandas code
                try:
                    tI = pd.to_numeric(tI)
                except:
                    pass # 250123: set to match prev behavior where we passed error='ignore' above, we may want to check why there are errros in future
                    #warnings.warn('some kind of error on data file parsing, should look at it and decide if ignore warning')
            # now sort into: consts, len 0.  dfD: len nTrials.  miscD: all else
            # misc fields first
            itemLen = containers.lenorzero(tI)
            if itemLen == 0 or tN in [
                "startDateVec",
                "savedDataName",
                "trPer80V",
                "block2TrPer80V",
                "constList",
            ]:
                constD[tN] = tI
            elif itemLen == nTrials:
                dfD[tN] = tI
            else:
                miscD[tN] = tI
        # print(miscD.keys())

        # turn dfD into a dataframe and return

        df = pd.DataFrame(dfD)
        # small sanity checks
        assert self.subjectNum == in0["subjectNum"].item()
        return (df, miscD, constD)


def file_name_data(subj, datestr):
    return "data2-i%04d-%6s.mat" % (subj, datestr)


def file_name_pdf(subj, datestr):
    return "%6s-behav2-ii%04d.pdf" % (subj, datestr)
