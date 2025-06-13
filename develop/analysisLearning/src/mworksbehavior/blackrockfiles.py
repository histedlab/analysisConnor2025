import numpy as np
import pandas as pd
import collections
import logging

import neo  # the blackrock read library

a_ = np.asarray
r_ = np.r_

from . import mwkfiles

log = logging.getLogger(__name__)

_mwEventNT = collections.namedtuple("_mwEventNT", "codename value mwTimestampUs brTimestampS")


class BlackrockFileWithSerial:
    """Read a blackrock file and parse its codes.
    BR file must have digital events and serial events

    Requires the neo library (for now, can be converted to support brPy later)

    Important fields:
        - brName (str), filename
        - brIO (neo BlackrockIO object)
        - brSegment (neo)
        - brDigEventTs, brDigEventCodes - vectors, strobed words from br digital events
        - mwEncodedTrialTsUs - mw Us value of trial start timestamp, decoded from BR dig event stream
        - brSerialTs, brSerialCodes - vectors from br serial event stream
        - mwEndStateDf - DataFrame, end state  of each mw trial, decoded from br serial
        - mwEventDf - Dataframe, event stream from mworks, decoded from br serial



    Notes:
        - All trial numbers start at zero!  (first trial is trial 0)
        - Eventually should be converted to a mixin that can be used with the MWK file classes above.
        - As with all our classes we should do as much parsing/setup as possible in __init__()
        - May at some point want to support multiple segments, now only does one"""

    def __init__(self, brName, tryToFixFile=False):
        self.brName = brName

        self.brIO = neo.io.BlackrockIO(brName)
        self.brSegment = self.brIO.read_segment()

        br = self.brIO  # shortcuts
        brs = self.brSegment

        # read digital events from blackrock
        evs = brs.events
        eN = np.flatnonzero(a_([ev.name == "digital_input_port" for ev in evs]))[0]

        # check for neo errors
        lens = [len(ev) for ev in evs]
        if np.allclose(lens, lens[0], rtol=0.1, atol=1):
            # for our files, one event len should be short (digital), one long (serial)
            # bad neo versions return serial, and dig and serial together.
            # to fix: install histedlab neo version, or wait for their fix to propagate
            raise RuntimeError(
                "Looks like bad neo event return: using correct neo version? Install histedlab neo."
            )

        # for iE,ev in enumerate(evs):
        #    print('ev %d, name %s, len %d' %(iE,ev.name,len(ev)))
        # print('channel is %d' % eN)
        self.brDigEventTs = a_(evs[eN].times)
        e0 = evs[eN].labels
        self.brDigEventCodes = a_([int(x) for x in e0])

        # parse digital event codes into trials
        (self.brDigTrCodeL, self.mwEncodedTrialTsUs, discardedD) = mwkfiles.parse_digital_stream(
            self.brDigEventCodes, self.brDigEventTs, tryToFixFile=tryToFixFile
        )

        # read serial events from blackrock
        eN = np.flatnonzero(a_([ev.name == "serial_input_port" for ev in evs]))[0]
        self.brSerialTs = a_(evs[eN].times)
        e0 = evs[eN].labels
        self.brSerialCodes = a_([int(x) for x in e0])
        # parse into trials
        (self.mwEndStateDf, self.mwEventDf, self._mwCodec) = mwkfiles.parse_blackrock_serial_stream(
            self.brSerialCodes, self.brSerialTs
        )

        # some misc computations and checks and potential fixups
        self.nTrials = int(np.max(self.mwEventDf.trialNum) + 1)
        if self.nTrials != len(self.brDigTrCodeL):
            # if serial has one more trial than digital, it's possible
            # a trial at the beginning lacks start codes but the serial was sent. Try to fix
            # by dropping first serial trial.
            if tryToFixFile and (self.nTrials == len(self.brDigTrCodeL) + 1):
                log.warning("One extra MWorks trial found, trying to fix by dropping first serial trial")
                keepIx = self.mwEventDf.trialNum > 0
                self.mwEventDf = self.mwEventDf.loc[keepIx, :]
                self.mwEventDf.trialNum = self.mwEventDf.trialNum - 1
                self.nTrials = self.nTrials - 1
                self.mwEndStateDf = self.mwEndStateDf.iloc[1:, :]
            else:
                raise mwkfiles.CorruptFileError("Error: number of trials in dig and serial stream differs")

        # do alignment of mwEvents to blackrock timing
        self.mwEventDf = self._compute_blackrock_timing_for_mwevents()

    def _compute_blackrock_timing_for_mwevents(self):
        """Adds a brTimestampS column to mwEventDf.
        Uses last start code (of 3) to do alignment.
        Also removes any codes before the last start from mwEventDf.
        Returns:
             new mwEventDf
        """

        outL = []
        for iT in range(self.nTrials):
            trIx = self.mwEventDf.trialNum == iT
            trDf = self.mwEventDf.loc[trIx, :]

            # find trial start mw timestamp
            desN = np.flatnonzero(
                (trDf.codename == "strobedDigitalWord") & (trDf.value == mwkfiles._partdigcodec.start)
            )
            # chop the trial df from the last run of start codes
            mwTs = trDf.mwTimestampUs.iloc[desN[-1]]  # last start code
            trDf = trDf.iloc[desN[-3] :, :]  # truncate up to first start code
            assert trDf.value.iloc[0] == mwkfiles._partdigcodec.start

            # find br timestamp
            trCodes = self.brDigTrCodeL[iT]
            desN = np.flatnonzero(trCodes == mwkfiles._partdigcodec.start)
            brTs = trCodes.index[desN[-1]]  # last start code
            # print(mwTs,brTs)

            # adj all mw timestamps to match br
            trDf = trDf.assign(brTimestampS=(a_(trDf.mwTimestampUs, dtype="f8") - mwTs) / 1e6 + brTs)
            outL.append(trDf)

        mwEventDf = pd.concat(outL, ignore_index=True)

        # checks
        mwTsS = mwEventDf.mwTimestampUs / 1e6
        mwBrDiffS = mwTsS - mwEventDf.brTimestampS
        if np.ptp(mwBrDiffS) > 0.020:
            # 180713 test file has 6ms, so leave some cushion
            raise mwkfiles.CorruptFileError("Max MW/computed BR timestamp diff is greater than 20ms: bug?")

        return mwEventDf

    def _get_mw_event_num(self, trialNum, codename, occurrence=0, value=None):
        desNs = np.flatnonzero((self.mwEventDf.trialNum == trialNum) & (self.mwEventDf.codename == codename))
        if value is not None:
            desIx = self.mwEventDf.value.iloc[desNs] == value
            desNs = desNs[desIx]
        if len(desNs) == 0:
            raise RuntimeError(
                "codename %s not found in trial %d (value restrict: %s)" % (codename, trialNum, value)
            )
        else:
            return desNs[occurrence]

    def get_mw_event(self, trialNum, codename, occurrence=0, value=None):
        """
        Args:
            occurrence: can be 0, -1 etc - as indexing; must be a scalar
            trialNum: 0-origin, can be a vector, if None means all trials
            value: if value is None, ignore.  If not, match against it - return only codes that match value

        Returns: _mwEventNT
        """
        if trialNum is None:
            trialNum = np.arange(self.nTrials)
        if not hasattr(trialNum, "__len__"):  # is scalar
            trialNum = [trialNum]
        desNL = []
        for (iT, tT) in enumerate(trialNum):
            desNL.append(self._get_mw_event_num(tT, codename, occurrence, value))

        return self.mwEventDf.iloc[desNL, :]
