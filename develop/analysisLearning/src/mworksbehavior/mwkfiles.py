from __future__ import print_function

"""Read mwk files, convert to our dataframe (mwk-df) format, either .h5 or .pkl.gz ext, process the results.

Classes:
    MWKFile:  root class for analyzing data from our experiments.  All expt-specific classes descend from this class
    RetinotopyMap1MWKFile:  for RetinotopyMap1; used now for imaging analysis.  (Descends from MWKFile)

Notes:
    All possible processing should be done when a class is instantiated

Todos:
    Move the digcodec etc. constants to this file.

171116 MH: created, moved from Intrinsic_Imaging repository
"""

import pandas as pd
import os
import numpy as np
import logging
import gzip
import json
import io
import collections
import codecs
import warnings
import types
from pathlib import Path  # only python >= 3.4

r_ = np.r_
a_ = np.asarray

log = logging.getLogger(__name__)


class CorruptFileError(Exception):
    pass


# eventually we may wish to get this from Expt package, but for now this avoids adding a dependency.
_partdigcodec = pd.Series({"start": 170, "end": 85, "trialTimestampStart": 200, "trialTimestampEnd": 201})
_experimentXmlTrialIds = {  # taken from ExperimentXML/PythonBridgeCode/Expt/constants.py on 211014
    "HoldAndDetectConstant8": 8,
    "HoldAndDetectConstant5": 10,
    "DirTuningMapping": 11,
    "OldChRMappingTwelve": 12,  # deprecated
    "OldChRMappingTen": 13,  # deprecated
    "DualPress": 14,
    "ContrastMap1": 15,
    "RetinotopyMap1": 16,
    "AuditoryMap1": 17,
    "AuditoryFluoMap1": 18,
    "ChRMapping": 19,
    "HoldAndDetectConstant19": 21,  # number clashes: 19!=21, but oh well.  Add one so it's a bit further away
    "RetinotopyMap2": 22,
    "RetinotopyMap2Stim": 23,
}


class MWKFile:
    """Read a MWK file, parse data in any general ways that are useful for all experiments.
    To specialize this for particular experiments, subclass it.

    Attributes:
        df
        filename
        basename
        dfEncodes

    Notes:
        Try to have all parsing done on object creation.  To do that in child classes, override __init__() and call your parsing functions
        from inside the child class's __init__().

    TODO:
        add a function to pull out codes by trial (to do this we ll need to parse code starts,
        probably calling parse_digital_stream()

    When moving to mworksbehavior:
        - convert below codec to use constants"""

    def __init__(
        self,
        filename,
        firstTrStartTimeUs=None,
        discardAfterTimeUs=None,
        discardBeforeTimeUs=None,
        doTryFixCorrupt=False,
        fixDfFn=None,
        confirmExptXmlIdL=None,
    ):
        """Opens the file, saves all code info.
        Args:
            filename: can be mwk2 file or mwk-df (.h5/.pkl.gz) file (from mwk_to_h5)
                As of 2021 we no longer run tests for the old mwk files
            discardAterTimeUs:
            discardBeforeTimeUs:
            doTryFixCorrupt: apply heuristics to corrupt files and try to drop things to fix them
               - definitly not guaranteed to work but worth a try
            fixDfFn: a callable: fixedDf = fixDfFn(mwk.df) - run after code df is run, can be used
                to repair a broken code sequence in a specific way
            confirmExptXmlIdL: a scalar or list of possible exptXmlId values in the mwk2 file - set this to confirm
                this mwkfile analysis code is being used on the correct expt.

        """

        self.filename = filename
        self.basename = os.path.basename(filename)

        # check for input file, convert to h5 if needed
        inputP = Path(filename)
        if inputP.suffix == ".h5" or inputP.suffixes[-2:] == [".pkl", ".gz"]:
            readP = inputP  # these can be read directly
        elif inputP.suffix == ".mwk2":
            h5P = inputP.with_suffix(".h5")
            if h5P.is_file():
                log.info(f"Found h5 file corresponding to mwk file {inputP}")
                readP = h5P
            else:
                log.warning(f"No h5 file found for mwk2, creating it... ")
                from . import mwk_io  # lazy-load this as it only runs on macs with MWorks installed

                mwk_io.mwk_to_h5(inputP, h5P, keep_system_vars=True)
                log.warning(f"Done: {h5P} created.")
                readP = h5P
        else:
            raise RuntimeError(f"Input file not recognized (not h5 or mwk2): {inputP}")

        self.doTryFixCorrupt = doTryFixCorrupt

        # read h5 or the old pkl.gz format
        if readP.suffix == ".h5":
            df0 = pd.read_hdf(readP)
        elif readP.suffixes[-2:] == [".pkl", ".gz"]:
            df0 = pd.read_pickle(readP)
        else:
            raise RuntimeError(f"Unknown MWKFile extension: {ext}")

        self.df = df0.sort_values("timeUs").reset_index(drop=True)
        if fixDfFn is not None:
            log.warning(
                f"** Calling fixDfFn, try to repair codes/events: Ask if you do not know what this means!"
            )
            self.df = fixDfFn(self.df)

        df = self.df  # alias for ease

        # add a few extra fields
        self.df["timeFmStUs"] = self.df.timeUs - self.df.timeUs[0]

        if discardAfterTimeUs is not None:
            self.df = self.df.loc[self.df.timeUs <= discardAfterTimeUs, :]
        if discardBeforeTimeUs is not None:
            self.df = self.df.loc[self.df.timeUs > discardBeforeTimeUs, :]
        if len(self.df) == 0:
            raise RuntimeError("All codes removed by restriction")

        # some extra initial computations
        self.dfEncodes = df.loc[df.tagname == "strobedDigitalWord", :].copy()

        # check/enforce exptXmlId
        if (
            confirmExptXmlIdL is not None and not self.doTryFixCorrupt
        ):  # skip this check if doTryFixCorrupt is true
            allIds = self.df.loc[df.tagname == "experimentXmlTrialId", "value"].to_numpy()
            if len(allIds) == 0:
                raise CorruptFileError(
                    f"No ExptXmlTrialIds found in the file: looking for {confirmExptXmlIdL}"
                )
            unqIds = np.unique(allIds)
            if len(unqIds) > 1:
                raise CorruptFileError(
                    f"More than one exptXmlTrialIds found in file: {unqIds}, looking for {confirmExptXmlIdL}"
                )
            else:
                idInFile = unqIds[0]
                if not np.isin(idInFile, confirmExptXmlIdL):
                    raise CorruptFileError(
                        f"Looking for exptXmlTrialIds {confirmExptXmlIdL}, but found {idInFile}: correct MWKFile object for analysis?"
                    )

        self._parse_trial_times(firstTrStartTimeUs=firstTrStartTimeUs)

    def _parse_trial_times(self, firstTrStartTimeUs=None):
        """Look at digital encodes and find first trial start and last trial end.

        Args:
            firstTrStartTimeUs: None (default): autodetect.  Normally this is fine, you may need to set this manually to fix bad files

        Returns:
            nothing, but sets self.firstTrialStartTimeUs, self.firstTrialEndTimeUs, self.lastTrialEndTimeUs.

        TODO: should add ability to set:
            self.trialStartTimesUsL, self.trialEndTimesUsL - lists that have start and end for each trial"""
        dfEn = self.dfEncodes  # shorthand
        # print(self.dfEncodes.iloc[:210,:])

        # remove start and end codes inside trial start timestamp runs
        desRowNs = np.nonzero(dfEn.value.to_numpy() == _partdigcodec.trialTimestampStart)[0]
        dropNs = np.hstack([desRowNs + c for c in [1, 2, 3, 4]])  # 4 codes after 200
        dfEn = dfEn.drop(dfEn.index[dropNs])

        # first end code run
        firstStTs = dfEn.loc[dfEn.value == _partdigcodec.start, :].timeUs.iloc[
            0
        ]  # look after at least one start to handle truncated tr at start
        dfEndAll = dfEn.loc[(dfEn.value == _partdigcodec.end) & (dfEn.timeUs >= firstStTs), :].copy()
        dfEnd3 = dfEndAll.iloc[:3, :].copy()
        # print(dfEnd3)
        dfEnd3.reset_index(inplace=True)

        if (dfEnd3.timeUs.iloc[-1] - dfEnd3.timeUs[0]) > 5 * 1e3:
            if self.doTryFixCorrupt:
                # try using just the first value as the end of the first trial, maybe one code was dropped
                log.warning(
                    "doTryFix: First run of 3 end codes lasts too long: using endcode[0] as 1st trial end"
                )
                pass  # will be set below
            else:
                raise CorruptFileError("error: first run of 3 end codes lasts longer than 5 ms")
        self.firstTrEndTimeUs = dfEnd3.timeUs[0]

        # find first trial start
        dfStart3 = dfEn.loc[
            (dfEn.timeUs <= self.firstTrEndTimeUs) & (dfEn.value == _partdigcodec.start), :
        ].iloc[-3:, :]
        # print(dfStart3)
        dfStart3.reset_index(inplace=True)
        if firstTrStartTimeUs is not None:
            # use specified start time
            self.firstTrStartTimeUs = firstTrStartTimeUs
        else:
            runDiffUs = dfStart3.timeUs.iloc[-1] - dfStart3.timeUs.iloc[0]
            if runDiffUs > 5 * 1e3:
                if self.doTryFixCorrupt:
                    # just continue, we use the first code below.
                    log.warning(
                        f"doTryFix: First run of 3 start codes lasts too long ({runDiffUs/1e3:.3g}ms), but continuing"
                    )
                else:
                    raise CorruptFileError(
                        "error: first run of 3 start codes before first end lasts longer than 5 ms"
                    )
            self.firstTrStartTimeUs = dfStart3.timeUs.iloc[0]

        # find last trial end
        dfEndLast3 = dfEndAll.iloc[-3:, :].copy()
        self.lastTrEndTimeUs = dfEndLast3.timeUs.iloc[-1]


    def get_codedf(self, tagname, only_in_startend_times=True, value=None):
        """
        Get a subset of codes, and add a time difference column

        Args:
            tagname: can be string (single code to return) or sequence of strings.
            only_in_startend_times: default True
            value: None for all values, otherwise a list or single value to match
        """

        if type(tagname) == str:
            tagname = [tagname]
        desIx = np.in1d(self.df.tagname, tagname)
        if only_in_startend_times:
            desIx = (
                desIx & (self.df.timeUs >= self.firstTrStartTimeUs) & (self.df.timeUs < self.lastTrEndTimeUs)
            )
        if value is not None:
            try:
                value[0]
            except TypeError: # not a sequence
                value = [value]
            desIx = desIx & (np.in1d(self.df.value, value))
        b0 = self.df.loc[desIx, :].copy().sort_values("timeUs")
        b0["timeDiffUs"] = np.hstack((np.nan, b0.timeFmStUs[1:].values - b0.timeFmStUs[:-1].values))
        b0.reset_index(inplace=True)
        b0.rename(columns={"index": "df_index"}, inplace=True)
        if len(tagname) == 1:
            b0.drop("tagname", axis="columns", inplace=True)
        return b0

    def get_const_values(self, constNameL=[]):
        """Get variable values that should be constant for the whole file.

        Use last value found in file, usually set on first trial.

        Later/TODO: find values of all constants (in self.constS), not just the ones we specify

        Returns:
            constS: pd.Series of constant names (index) and values
        """

        cS = pd.Series(dtype="float64")  # Series, names as index levels
        if type(constNameL) == str:
            constNameL = [constNameL]
        for c in constNameL:
            tV = self.df.loc[self.df.tagname == c, "value"].iloc[-1]  #
            cS.loc[c] = tV
        return cS


class CounterStimMixin:
    """Contains methods that are useful for any experiments with counter-driven stim"""

    def parse_counted_stim(
        self,
        stims_to_keep="all",
        use_counter_as_start=True,
        stim_set_when="after_last_pre_count",
        counterVar="counterFIO",
        levelVar="gratingPySelLevel",
        extraVarL=["tGratingAzimuthDeg", "tGratingElevationDeg"],
        missing_extra_ok=False,
        do_try_to_fix=False,
    ):

        """Find stim variables and values on each counter trial.

        Must be called on __init__().

        Note: we keep stim only after firstTrStartTimeUs.

        Args:
            stims_to_keep: sequence: which grating stim to retain; drop others.  Used for fixing up bad files
            use_counter_as_start: use time counter goes to 1 as the start of the trial.  This is to workaround a bug
                    where start of trial codes (dig=170) sometimes get lost.  See 171117-debug-code-1230-2p-stim2-16x.ipynb
            stim_set_when: string.  Only does something with use_counter_as_start==True
            ￿    'after_last_pre_count': default, means the stim is set(chosen) right before the stim is turned on.
                    This is the way most expts worked, including RetinotopyMap1
                'before_first_pre_count': stim set (chosen) at start of trial
                    RetinotopyMap2Stim.mwel works this way
            counterVar: tagname (in MWK codec) of the counter variable
            levelVar: the selection variable set each trial
            extraVarL: a list of variables to get values for on each trial.  (Saved to self.stimDf)
            do_try_to_fix: as elsewhere, try to fix up files with problems


        Returns:
            nothing
        Sets: self.stimDf, self.counterVar, self.levelVar

        """

        ## NOTES 250119: this counter parsing code is all pretty fragile and complicated.
        # Summary: This needs to find when the stim level is set (stimPySelLevel) for each trial.
        # It also finds the place to start looking for stim level using either the counter tick pattern or the
        # trial start time, set before this is called.
        # Counter tick gets set to zero on reset in MWorks, at first press of start. There are either three
        # zero values before it or three nonzero values that come from the previous trial.
        # With the ngfriedman/2310/imaginggate-laseroff changes, the counter zero sometimes is quickly followed
        # by a one and sometimes there is no zero as if the one value is read before the zero can be.
        # Also, the timing of the zero/one is now fuzzy and can be delayed by a few ms, sometimes overlapping with
        # strobed=170 at trial start, or with stimPySelLevel.
        #
        # For future:
        # we should probably drop finding the start time from the counter tick, and just do it from trial start
        # Other than that, this code is as simple as it can be, I think.

        assert stims_to_keep is not None, "stims_to_keep should be 'all', not None"
        df = self.df

        # make sure the counter stream start is as expected.
        # if MWorks is not restarted, the first three counter values will have the old max value
        countIx = df.tagname == counterVar
        countDf = df.loc[countIx,:] # indices are the indices into df

        ## 241226: WORKAROUND for MWorks bug.
        # In some cases on rig C2, when scanimage is started, resetting the MWorks counter variable sets the counter to 1.
        # See MH LabArchives lab notebook, fall 2024, for details.
        # To address: we detect this situation and subtract 1 from all counter values.
        # Key check: if first 1 (in 4th or 5th index spot) follows previous code by < 1ms, it means the zero code was missed and
        # the first 1 is really zero. So subtract
        did_change_counter_values = False
        firstOneN = (countDf.value == 1).to_numpy().nonzero()[0][0]
        zeroToOneUs = countDf.timeUs.iloc[firstOneN] - countDf.timeUs.iloc[firstOneN-1] # this can be a nonzero
        if zeroToOneUs < 15000: # 15 ms threshold
            didSub=1
            warnings.warn(f"Found first counter tick to be 1, expected zero; subtracting 1 from all counter vals, altering df values")
            self.df.loc[countDf.index[firstOneN:], 'value'] = self.df.loc[countDf.index[firstOneN:], 'value'] - 1 # subtract 1
            firstOneN = firstOneN + 1
            did_change_counter_values = True
        # also, always drop the counter 0 code - it now sometimes occurs after the first 170 strobe due to the odd extra delayed
        # tick behavior.
        self.df.drop(countDf.index[firstOneN-1], axis='index', inplace=True)
        ## end workaround


        ## here we find the time to start looking for the first stim level (stimPySelLevel value), startUs ###########

        if use_counter_as_start:
            ## sets counterOneUs
            desIx = (df.tagname == counterVar) & (df.value == 1)
            # check for no counter 1 values, or more than 1
            if np.sum(desIx) == 0:
                raise CorruptFileError("counter var (%s) == 1 not found: error in file" % counterVar)
            elif np.sum(desIx) > 1:
                # tIx = df.tagname == counterVar
                # display(df.loc[tIx, :])
                warnings.warn(
                    "Found two counter values == 1 in file. File restarted? Dropping counter vals before last 1 to fix. "
                )
                # if there was an accidental start at end, you'll need to fix it up manually, or add an option here to do it.
                counterOneUs = df.timeUs[np.flatnonzero(desIx)[-1]]
            elif np.sum(desIx) == 1:
                tD = df.timeUs[desIx]
                counterOneUs = tD.iloc[0]

            ## set startUs based on counterOneUs
            if stim_set_when == "after_last_pre_count":
                startUs = counterOneUs
            elif stim_set_when == "before_first_pre_count":
                # find the levelVar right before the 1 value
                desIx = (df.tagname == levelVar) & (df.timeUs < counterOneUs)
                assert np.sum(desIx) > 0, "no level values found before counter=1?"
                desN = np.nonzero(desIx.to_numpy())[0][-1]  # last one
                startUs = df.timeUs[desN]  # epsilon (10us) past the counter zero code

            else:
                raise RuntimeError("bad value %s for stim_set_when" % stim_set_when)

        else:
            # take startUs from the first trial start time
            startUs = self.firstTrStartTimeUs

        # er#ror check: check startUs by checking the counter values
        tV = df.value[(df.tagname == counterVar) & (df.timeUs >= startUs)
                      & (df.timeUs < self.lastTrEndTimeUs)]
        if (tV == 0).any():
            raise CorruptFileError("counter zero tick found, counter ticks should start with 1")
        nCounterTicks = len(np.unique(tV))
        if not nCounterTicks == np.max(tV):
            raise CorruptFileError(
                "Not all counter vals betw %d and %d present, found %d: double-counting/skipping? check."
                % (np.min(tV), np.max(tV), nCounterTicks)
            )

        ## extract all the desired variable values after startUs ##############

        getVarL = [levelVar] + extraVarL
        outD = {}
        for v in getVarL:
            df0 = df.loc[(df.tagname == v) & (df.timeUs >= startUs)
                         & (df.timeUs < self.lastTrEndTimeUs), :].copy()
            df0.reset_index(inplace=True)
            if len(df0) == 0:
                if missing_extra_ok:
                    continue  # just ignore it if not in file: allows us to retain back compat if we add new vars
                else:
                    raise RuntimeError(f"did not find variable {v} in file: bug?")

            if type(df0.value[0]) == str:
                pass  # leave strings unconverted
            else:
                # try to convert everything else to float.  If errors, add exceptions above, or provide options to fix file
                outD[v] = pd.to_numeric(df0.value, errors="coerce")  # or set error=ignore?
        self.stimDf = pd.DataFrame(outD)

        # df2 = df.loc[(df.tagname == "gratingPySelLevel") & (df.timeUs >= 0), :].copy()
        #if drop_last_stim:
        #    self.stimDf = self.stimDf.iloc[:-1, :].copy()
        if stims_to_keep != "all":
            self.stimDf = self.stimDf.iloc[stims_to_keep, :].copy()

        ## final error checks ###############
        # check we have the correct number of ticks
        nTotalStims = len(self.stimDf)
        if not nCounterTicks % nTotalStims == 0:
            if do_try_to_fix and did_change_counter_values:
                if (nCounterTicks+1) % nTotalStims == 0:
                    warnings.warn('Trying to fix file: Number of frames is one less than expected, probably first trial is short, be careful w/ analysis')
            else:
                raise CorruptFileError(f'Number of total stimuli {nTotalStims} does not divide total' 
                                       f'counter ticks {nCounterTicks} equally: error in counter or stim?')

        ## assign some informational fields and exit #############

        self.counterVar = counterVar
        self.levelVar = levelVar
        self.counterStats = types.SimpleNamespace(nCounterTicks=nCounterTicks, nTotalStims=nTotalStims,
                                                  ticksPerStim=nCounterTicks/nTotalStims)


    def save_stim_params(self, stimoutname):
        self.stimDf.to_hdf(stimoutname, key="stimsByTrial", mode="w")
        print("Done, written to %s" % stimoutname)

    def _compute_imaging_constants_preposttrig(
        self,
        counterVar="counterFIO",
        levelVar="gratingPySelLevel",
        preVar="counterNPre",
        duringVar="counterNStim",
        postVar="counterNPost",
    ):
        """Find a subset of counter constants useful for imaging analysis.
        Args:
            duringVar: can be None


        Returns:
            none, but sets self.constS, self.levels, self.nstim, self.nframes_stim, self.nreps
                self.preVar, self.duringVar, self.postVar
        """

        tdf = self.get_codedf(counterVar)
        self.firstCounterUs = tdf.timeUs[tdf.value == 1]
        assert len(self.firstCounterUs)>0, 'No first counter tick found'
        self.preVar = preVar
        self.duringVar = duringVar
        self.postVar = postVar

        # compute constS
        constL = [preVar, postVar]
        if duringVar is not None:
            constL.append(duringVar)
        self.constS = self.get_const_values(constL)
        cS = self.constS

        self.levels = np.unique(getattr(self.stimDf, levelVar))
        self.nstim = len(self.levels)
        self.nframes_stim = getattr(cS, preVar) + getattr(cS, postVar)
        if duringVar is not None:
            self.nframes_stim += getattr(cS, duringVar)

        (frac, self.nreps) = np.modf(len(self.stimDf) * 1.0 / self.nstim)
        if not np.isclose(frac, 0):
            raise CorruptFileError(
                "Error/bug: number of total stim %d does not evenly divide number of unique stimuli %d"
                % (len(self.stimDf), self.nstim)
            )

    def drop_artifact_frames(self, nStimArtifactFrs=1):
        """Edit the mwf file and remove count of artifact frames.

        Args:
            nStimArtifactFrs: default 1.  If zero, don't change the mwf obj

        """
        print("Dropping %d artifact frames from mwf data structures" % nStimArtifactFrs)
        nTotFr = self.nreps * self.nstim * self.nframes_stim
        print("Before artifact drop: nStims: %d, nReps: %d, totalFr %d" % (self.nstim, self.nreps, nTotFr))
        # sub nStimArtifactFrs from pre and nrames_stim
        setattr(self.constS, self.preVar, getattr(self.constS, self.preVar) - nStimArtifactFrs)
        self.nframes_stim -= nStimArtifactFrs
        nTotFr = self.nreps * self.nstim * self.nframes_stim
        print("After artifact drop:  nStims: %d, nReps: %d, totalFr %d" % (self.nstim, self.nreps, nTotFr))


class ChRMap1MWKFile(CounterStimMixin, MWKFile):
    def __init__(
        self,
        mwk_name,
        firstTrStartTimeUs=None,
        discardAfterTimeUs=None,
        discardBeforeTimeUs=None,
        stims_to_keep="all",
        use_counter_as_start=False,
        doTryFixCorrupt=False,
    ):
        """
        Args:
            stims_to_keep: see CounterStimMixin
            firstTrStartTimeUs, discardAfterTimeUs, doTryFixCorrupt: see MWKFile


            note: use_counter_at_start param must be true: often tLaserPySelLevel is set before counterFIO is 1.


        """

        MWKFile.__init__(
            self,
            mwk_name,
            firstTrStartTimeUs,
            discardAfterTimeUs=discardAfterTimeUs,
            discardBeforeTimeUs=discardBeforeTimeUs,
            doTryFixCorrupt=doTryFixCorrupt,
            confirmExptXmlIdL=_experimentXmlTrialIds["ChRMapping"],
        )

        self.parse_counted_stim(
            stims_to_keep=stims_to_keep,
            use_counter_as_start=use_counter_as_start,
            stim_set_when="after_last_pre_count",
            counterVar="counterFIO",
            levelVar="tLaserPySelLevel",
            extraVarL=[
                "tAPeakPowerMw",
                "tABaselinePowerMw",
                "tARampTrain",
                "tATrainPeriodMs",
                "tATrainNPulses",
                "tATrainPulseLengthMs",
                "tARampLengthMs",
                "tARampExtraConstantLengthMs",
                "tAStartOffsetMs",
                "tBPeakPowerMw",
                "tBBaselinePowerMw",
                "tBRampTrain",
                "tBTrainPeriodMs",
                "tBTrainNPulses",
                "tBTrainPulseLengthMs",
                "tBRampLengthMs",
                "tBRampExtraConstantLengthMs",
                "tBStartOffsetMs",
            ],
        )

        def checkc(x):
            assert np.all(
                x == x.iloc[0]
            ), "within a selection level, constants are different: check.  Bug in mwk file?"
            return x.iloc[0]

        self.levelDf = self.stimDf.groupby(self.levelVar).agg(checkc)

    def compute_imaging_constants(self):
        """Find a subset of retinotopy constants useful for imaging analysis.

        Returns:
            none, but sets self.constS, self.levels, self.nstim, self.nframes_stim.
        """

        self._compute_imaging_constants_preposttrig(
            counterVar="counterFIO",
            levelVar="tLaserPySelLevel",
            preVar="counterNPreTrig",
            duringVar=None,
            postVar="counterNPostTrig",
        )


class RetinotopyMap1MWKFile(CounterStimMixin, MWKFile):
    def __init__(
        self,
        mwk_name,
        firstTrStartTimeUs=None,
        discardAfterTimeUs=None,
        stims_to_keep="all",
        use_counter_as_start=True,
        fixDfFn=None,
    ):
        """
        Args:
            stims_to_keep: sequence: which grating stim to retain; drop others.  Used for fixing up bad files
            use_counter_as_start: use time counter goes to 1 as the start of the trial.  This is to workaround a bug
            where start of trial codes (dig=170) sometimes get lost.  See 171117-debug-code-1230-2p-stim2-16x.ipynb
        Notes:
            - supports RetinotopyMap1 and RetinotopyMap2.mwel.  Note that as of 190531, tGrating* are renamed
            to tStim* in RM1 files to match new names in RM2.
        """

        MWKFile.__init__(
            self,
            mwk_name,
            firstTrStartTimeUs,
            discardAfterTimeUs,
            fixDfFn=fixDfFn,
            confirmExptXmlIdL=[
                _experimentXmlTrialIds["RetinotopyMap1"],
                _experimentXmlTrialIds["RetinotopyMap2"],
            ],
        )

        # auto-detect version/experiment ID
        varA = np.unique(self.df.tagname)
        if "gratingPySelLevel" in varA:
            self.exptIdStr = "RetinotopyMap1"
            levelVar = "gratingPySelLevel"
            extraVarL = ["tGratingAzimuthDeg", "tGratingElevationDeg"]
        elif "stimPySelLevel" in varA:
            self.exptIdStr = "RetinotopyMap2"
            levelVar = "stimPySelLevel"
            extraVarL = [
                "tStimAzimuthDeg",
                "tStimElevationDeg",
                "tGratingSpeedDps",
                "tGratingSpatialFreqCpd",
                "tOriNoiseF0Cpd",
                "tOriNoiseSigmaFCpd",
                "tOriNoiseSigmaThetaDeg",
                "tStimContrast",
                "tStimDirectionDeg",
                "tStimWidthDeg",
                "tStimHeightDeg",
                "tStimBaseDirectionDeg",
                "tStimBaseContrast",
            ]

        self.parse_counted_stim(
            stims_to_keep=stims_to_keep,
            use_counter_as_start=use_counter_as_start,
            stim_set_when="after_last_pre_count",   #for Intrinsic Imaging, RM1, or RM2 data, use “after_last_pre_count”
            #stim_set_when="before_first_pre_count",  #for retinotopy, RM2Stim data etc. using "before_first_pre_count"
            counterVar="counterFIO",
            levelVar=levelVar,
            extraVarL=extraVarL,
            missing_extra_ok=True,
        )
        # above sets: self.stimDf, self.counterVar, self.levelVar
        if self.exptIdStr == "RetinotopyMap1":
            self.stimDf = self.stimDf.rename(
                {
                    "tGratingAzimuthDeg": "tStimAzimuthDeg",
                    "tGratingElevationDeg": "tStimElevationDeg",
                    "gratingPySelLevel": "stimPySelLevel",
                },
                axis="columns",
            )
        self.levelVar = "stimPySelLevel"  # if it was gratingPySelLevel, this got changed above

    def compute_imaging_constants(self):
        """Find a subset of retinotopy constants useful for imaging analysis.

        Returns:
            none, but sets self.constS, self.levels, self.nstim, self.nframes_stim.
        """

        self._compute_imaging_constants_preposttrig(
            counterVar="counterFIO",
            levelVar="stimPySelLevel",
            preVar="counterNPre",
            duringVar="counterNStim",
            postVar="counterNPost",
        )

        if self.constS.counterNPost != 0:
            log.warning("normally we should use zero post frames: just build those into pre")


class RetinotopyMap2StimMWKFile(CounterStimMixin, MWKFile):
    def __init__(
        self,
        mwk_name,
        firstTrStartTimeUs=None,
        discardAfterTimeUs=None,
        stims_to_keep="all",
        use_counter_as_start=True,
        doTryFixCorrupt=False,
        fixDfFn=None,
    ):
        """
        Args:
            stims_to_keep: sequence: which grating stim to retain; drop others.  Used for fixing up bad files
            use_counter_as_start: use time counter goes to 1 as the start of the trial.  This is to workaround a bug
            where start of trial codes (dig=170) sometimes get lost.  See 171117-debug-code-1230-2p-stim2-16x.ipynb
            doTryFixCorrupt: passed to MWKFile
            fixDfFn: passed to MWKFile
        Notes:
            - supports RetinotopyMap2Stim.  RM1 and RM2 are supported by RetinotopyMap1MWKFile.
        """

        MWKFile.__init__(
            self,
            mwk_name,
            firstTrStartTimeUs,
            discardAfterTimeUs,
            doTryFixCorrupt=doTryFixCorrupt,
            fixDfFn=fixDfFn,
            confirmExptXmlIdL=_experimentXmlTrialIds["RetinotopyMap2Stim"],
        )

        # auto-detect version/experiment ID
        self.exptIdStr = "RetinotopyMap2"
        self.levelVar = "stimPySelLevel"

        # construct extraVarL
        extraVarL = []
        visVarL = [
            "tStimNAzimuthDeg",
            "tStimNElevationDeg",
            "tStimNGratingSpeedDps",
            "tStimNGratingSpatialFreqCpd",
            "tStimNOriNoiseF0Cpd",
            "tStimNOriNoiseSigmaFCpd",
            "tStimNOriNoiseSigmaThetaDeg",
            "tStimNContrast",
            "tStimNDirectionDeg",
            "tStimNWidthDeg",
            "tStimNHeightDeg",
            "tStimNBaseDirectionDeg",
            "tStimNBaseContrast",
        ]
        for tV in visVarL:
            extraVarL.append(tV.replace("StimN", "Stim1"))
            extraVarL.append(tV.replace("StimN", "Stim2"))
        laserVarL = [
            "tLaserXPeakPowerMw",
            "tLaserXBaselinePowerMw",
            "tLaserXRampTrain",
            "tLaserXTrainPeriodMs",
            "tLaserXTrainNPulses",
            "tLaserXTrainPulseLengthMs",
            "tLaserXRampLengthMs",
            "tLaserXRampExtraConstantLengthMs",
            "tLaserXStartOffsetMs",
        ]
        for tV in laserVarL:
            extraVarL.append(tV.replace("LaserX", "A"))
            extraVarL.append(tV.replace("LaserX", "B"))

        self.parse_counted_stim(
            stims_to_keep=stims_to_keep,
            use_counter_as_start=use_counter_as_start,
            stim_set_when="before_first_pre_count",
            counterVar="counterFIO",
            levelVar=self.levelVar,
            extraVarL=extraVarL,
            do_try_to_fix=doTryFixCorrupt,
        )
        # above sets: self.stimDf, self.counterVar, self.levelVar

    def save_grating_params(self, txt):
        """Backward compat: remove in a month or two 171212"""
        raise RuntimeError("Error: save_grating_params() is now named save_stim_params() -- change your code")

    def compute_imaging_constants(self):
        """Find a subset of retinotopy constants useful for imaging analysis.

        Returns:
            none, but sets self.constS, self.levels, self.nstim, self.nframes_stim.
        """

        self._compute_imaging_constants_preposttrig(
            counterVar="counterFIO",
            levelVar="stimPySelLevel",
            preVar="counterNPre",
            duringVar="counterNStim",
            postVar="counterNPost",
        )


def parse_digital_stream(codes, ts, tryToFixFile=False):
    """General function to parse any stream of digital codes, fm Blackrock, mworks, etc.
    Args:
        tryToFixFile [default False]: if true, attempt targeted fixup of broken files


    Returns:
        (trialCodeL, mwTrialTimestampUs, discardedCodeS)

        trialCodeL is list, len nTrials.  Each el is a pandas.Series with codes from that trial, index is timestamps.
        trTimestampV: decoded (int) 4-byte MWorks microsecond timestamp code
        discardedCodeS: pandas series with discarded codes.  Index is ts
    """
    DEBUG = False

    def dprint(s, end=None):
        if DEBUG:
            print(s, end=end)

    ts = a_(ts)
    codenums = a_(codes).copy()

    # first find and remove the trial timestamp digital codes
    mwTsD = {}
    stIx = codenums[0:-5] == _partdigcodec.trialTimestampStart
    endIx = codenums[5:] == _partdigcodec.trialTimestampEnd
    timestampStNs = np.nonzero(stIx & endIx)[0]
    didwarn = False
    for (iN, tN) in enumerate(timestampStNs):
        # print(tN, codenums[tN-3:tN+7])
        if tN > 0 and not (
            codenums[tN - 1] == _partdigcodec.start or codenums[tN - 2] == _partdigcodec.start
        ):
            # XML should always deliver timestamps after a start code, if not can mean mixing of serial/dig codes
            # sometimes the 200 code can be duplicated: ignore here
            raise CorruptFileError(
                "trial timestamp start code not preceded by start code (tr %d), corrupt file?" % iN
            )
        tCs = codenums[tN + 1 : tN + 5]
        dictKey = ts[tN]  # indexed by timestamp of code 200/trialTimestampStart
        # 4 byte stream is big-endian: biggest value is first, LS byte is last
        mwTsD[dictKey] = (
            codenums[tN + 1] * 2 ** 24
            + codenums[tN + 2] * 2 ** 16
            + codenums[tN + 3] * 2 ** 8
            + codenums[tN + 4]
        )
        if np.all(tCs == 0) and not didwarn:
            warnings.warn("MWTrialTimestamp zero found - error in sending across digital?  needs fix")
            didwarn = True
        codenums[tN + 1 : tN + 5] = 9999  # placeholder so we can drop outside for loop for speed

    # drop the 4-byte runs, leave the trialTimestamp start/end codes
    dropIx = codenums == 9999
    codenums = codenums[~dropIx]
    ts = ts[~dropIx]

    # now iterate over all codes and find trials
    trCodeL = []
    currTrStart = None
    discardedD = {}  # timestamps are ordered so regular dict is fine here, no OrderedDict() needed
    iC = 0  # we increment iCode manually before continue stmts below
    while True:
        if iC + 1 > len(codenums):
            break  # past the end, done
        tC = codenums[iC]
        tTs = ts[iC]
        # dprint('c %d ts %d' % (tC,tTs)) # debug

        # if not in a trial, look for start code, discard codes until found
        if currTrStart is None:
            if tC == _partdigcodec.start:  # found a trial start
                if tryToFixFile:
                    # find end of run to handle runs < 3 OR > 3.  Look 10 ahead, ignore end of file, wld be rare to collide
                    # fancy trick: w/ cumprod any True after the first false is zero/False
                    isStartCont = np.cumprod(codenums[iC + r_[0:10]] == _partdigcodec.start)
                    nStarts = np.flatnonzero(isStartCont)[-1] + 1
                    if nStarts != 3:
                        log.warning(
                            "in tr %d, ts %g, found %d instead of 3 start codes"
                            % (len(trCodeL) + 1, tTs, nStarts)
                        )
                else:
                    nStarts = 3
                    if not codenums[iC + nStarts] == _partdigcodec.start:
                        raise CorruptFileError("Start code found outside a run of 3. set tryToFixFile?")
                currTrStart = iC
                iC += nStarts
                continue
            else:
                discardedD[tTs] = tC  # discard this code, iterate
                iC += 1
                continue
        else:
            # in a trial, look for end or another start to find truncated trials
            if tC == _partdigcodec.start:
                # found another start before end
                if not codenums[iC + 2] == _partdigcodec.start:
                    import pdb

                    pdb.set_trace()
                    raise CorruptFileError("Start code found outside a run of 3.")
                discardNs = r_[currTrStart : iC - 1]  # discard up to this start run
                currTrStart = iC

                dprint(discardNs)
                dprint(codenums[discardNs])
                log.warning(
                    "Found start run during trial, truncating and discarding %d codes: %s"
                    % (len(discardNs), codenums[discardNs])
                )
                discardedD.update(dict(zip(ts[discardNs], codenums[discardNs])))
                iC += 3  # skip start run
                continue

            elif tC == _partdigcodec.end:
                if not codenums[iC + 2] == _partdigcodec.end:
                    raise CorruptFileError("End code found outside a run of 3")

                tTrNs = r_[currTrStart : iC + 3]
                tTrC = pd.Series(codenums[tTrNs], index=ts[tTrNs])

                dprint("Found full trial: ", end="")
                # dprint(tTrNs)
                dprint(codenums[tTrNs])
                dprint(len(tTrNs))
                # if len(trCodeL) > 1:
                #    break  # debug

                trCodeL.append(tTrC)

                currTrStart = None
                iC += 3
                continue

            else:
                # in trial, no end, just iterate
                iC += 1
                continue

    # now for every complete trial assign the mworks timestamp.
    # we do this after trial parsing to avoid corrupted trials getting timestamps assigned
    mworksTsV = np.zeros((len(trCodeL),))
    # print(np.sort(list(mwTsD.keys())))
    for (iT, tT) in enumerate(trCodeL):
        codeIx = tT == _partdigcodec.trialTimestampStart
        # subtle indexing here.  If 200 is duplicated, decoding code above will use last, as it finds st/end sep by 4.
        # so here we use last too to deal with dups.
        dictKey = tT.index.values[codeIx][-1]  # if len 0, will throw an error
        if dictKey not in mwTsD:
            if tryToFixFile:
                log.warning("missing timestamp found for tr %d: replacing with nan" % (iT))
                mworksTsV[iT] = np.nan
                # This can happen if 200 is dropped or one of the ts-encoding codes is dup'd
        else:
            mworksTsV[iT] = mwTsD[dictKey]

    dprint("n trials: %d" % len(trCodeL))
    dprint("nCodes first trial: %d" % len(trCodeL[0]))
    dprint(trCodeL[0])

    return (trCodeL, mworksTsV, pd.Series(discardedD))


def parse_blackrock_serial_stream(serCodes, serTs):
    """Convert a stream of serial codes from a blackrock data file to Python objects for analysis

    Returns:
        (mwEndStateDf, mwEventDf, codecS)
        codecS: pd.Series, index code nums, values codenames/strings

    Note: timestamps
    """
    # Notes:
    #  Format for each trial: 'GZipStartTrialDataGDATAGzipEndTrialData'
    #  GDATA is a gzip FILE, when uncompressed gives: 'JSONEndState\nJSONCodec\nJSONEventStream\n'
    # split the stream into trials, using big gaps in timestamp

    diffTs = np.diff(serTs)
    serStartNs = np.concatenate(([0], np.flatnonzero(diffTs > 0.1) + 1))  # >100ms
    trSerCodeL = []
    for (iS, tS) in enumerate(serStartNs):
        if iS == len(serStartNs) - 1:
            lastN = len(serCodes)
        else:
            lastN = serStartNs[iS + 1]
        desC = serCodes[tS:lastN]
        trSerCodeL.append(desC)
    del serTs  # we don't need the blackrock timestamps for each code beyond this point

    # iterate through list and decode each json segment
    serEndStateL = []
    serCodecL = []
    serEventsL = []
    for (iL, tL) in enumerate(trSerCodeL):
        # if we've screwed up parsing, or parsing into strings, these can be missing or corrupted.
        # If these errors raise, check the strings.
        if not codecs.decode(tL[0:18].astype("b"), "ascii") == "GzipStartTrialData":
            print(tL[0:18])
            print("string: %s" % codecs.decode(tL[0:20].astype("b"), errors="replace"))
            raise RuntimeError("At start of trial in serial: MWorks magic string not found.  Corrupt file?")
        if not codecs.decode(tL[-16:].astype("b"), "ascii") == "GzipEndTrialData":
            raise RuntimeError("At end of trial in serial: MWorks magic string not found.  Corrupt file?")

        desB = tL[18:-16].astype("b")
        outb = gzip.GzipFile("", "rb", fileobj=io.BytesIO(desB)).read()
        outStr = outb.decode("ascii")

        # break on newlines
        oL = outStr[:-1].split("\n")  # always a last newline; drop it here
        serEndStateL.append(json.loads(oL[0]))
        serCodecL.append(json.loads(oL[1]))
        serEventsL.append(json.loads(oL[2]))

    # ensure one codec
    for tC in serCodecL:
        if not tC == serCodecL[0]:
            raise RuntimeError("Codec changes: corrupt file?")
    codec = serCodecL[0]
    intKeys = np.asarray(list(codec.keys()), "uint16")
    codecS = pd.Series(dict(zip(intKeys, codec.values())))

    # convert into dataframe: end state
    dfcols = np.unique([list(s.keys()) for s in serEndStateL])
    endStateDf = pd.DataFrame(serEndStateL, columns=dfcols)

    # convert into dataframe: events
    tL = []
    for (iE, tE) in enumerate(serEventsL):
        dwIx = a_(tE["names"]) == "dW"
        tsCol = np.array(a_(tE["ts"])[~dwIx], dtype="uint64")
        # print(tsCol[0:100])
        # print(len(tsCol), len(np.unique(tsCol)))

        codeNums = a_(tE["names"])[~dwIx].astype("uint16")
        nameCol = pd.Series(codecS[codeNums].values, dtype="category")
        # print(np.flatnonzero(~dwIx))
        valsCol = a_(tE["values"], dtype="O")[~dwIx]
        trialCol = tsCol * 0 + iE
        tDf = pd.DataFrame(
            collections.OrderedDict(
                (("trialNum", trialCol), ("codename", nameCol), ("value", valsCol), ("mwTimestampUs", tsCol))
            ),
            index=np.arange(len(trialCol)),
        )

        tL.append(tDf)
    serialEventDf = pd.concat(tL, ignore_index=True)

    return (endStateDf, serialEventDf, codecS)
