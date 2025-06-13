# -*- coding: utf-8 -*-
"""Psychometric function manipulation: fits, plotting, etc.

Classes:
    ManyFitsFromParams:  does fits from param XLS, puts results in results XLS. 
        do_all_fits: iterates over paramsDf and does all fits, writes resultsDf and extraoutfiles
        do_all_plots: once do_all_fits is done, this can generate plots. 
        

Functions and attributes:
    weibullF: the callable that evalues the psych function.
    fit_and_bootstrap: takes a dataframe from mat_io and runs the weibull fit and bootstrap.
        Outside ManyFitsFromParams because it doesn't use the params/resultsDf
        _single_fit: subfunction of above
    

Notes:
    Common attributes of several functions:
        p: (double vector, len 3 or 4).  the param argument to the weibull function.
            Has 3 or 4 elements:  p[0] is threshold/position, p[1] slope, p[2] upper asymptote,
            p[4] lower asymptote

        do50PctThresh: bool.  If False, use pct=0.6321 for thresh (weibull CDF value at x=p[0])
            default: True

    Right now weibull is hardcoded; would be hard to change.  Ask MH if wanting other functions

Todo:
    - Add lower asymptote to Weibull
    - weight fits by number of trials


170822: MH, created
"""

import os
import scipy.io
import time
import pytoolsMH as ptMH
import pandas as pd
import numpy as np
import scipy
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import re
import warnings
import collections

nan = np.nan
from . import mat_io

a_ = np.array
r_ = np.r_

import logging

log = logging.getLogger(__name__)

debug = False


assert hasattr(pd.DataFrame(), "transform"), "Needs a later pandas version"


def df_makecols(df, columnL):
    """Make sure all columns exist in a dataframe, adding if missing.
    If no copy is needed by pandas, no copy is made here.

    Should eventually be moved to pytoolsMH somewhere"""
    cols = [x for x in columnL if x not in df] + df.columns.tolist()
    return df.reindex(columns=cols)


def weibullF(p, xs):
    """p[0] thresh; p[1] slope; p[2] upper asymp"""
    return p[2] * (1 - np.exp(-((xs / p[0]) ** p[1])))


def weibull_compute_thresh_from_p(p, do50PctThresh=True):
    """For Weibull, find threshold given parameters."""
    if do50PctThresh:
        thresh = p[0] * np.log(2) ** (1 / p[1])
        threshY = p[2] / 2
    else:
        thresh = p[0]
        threshY = 0.6321 * p[2]
    return (thresh, threshY)


def fit_and_bootstrap(trialDf, nBootstrapReps=None, intensName="intensVect"):
    """trialDf is a behav data dataframe from mb.mat_io
    Assumes len(trialDf) > 0"""

    # set up input DataFrame, some prep. data massaging
    df2 = trialDf.copy()
    df2 = df2.assign(isCorr=df2.trialOutcomeCell == "success", isFail=df2.trialOutcomeCell == "ignore")

    # do fit on real data
    truefit = _single_fit(df2, intensName=intensName)
    if truefit.invalidFit:
        ci95 = a_([nan, nan])
    else:
        # do bootstrap resampling and many fits
        ci = None
        if nBootstrapReps is not None:
            threshV = np.nan * np.zeros((nBootstrapReps,))
            for iR in range(nBootstrapReps):
                tDf = df2.sample(n=len(df2), replace=True)  # 401 µs per loop, 170820
                tF = _single_fit(tDf, intensName=intensName)
                threshV[iR] = tF.thresh
                # print(np.max(threshV))
            ci95 = (np.percentile(threshV, 2.5), np.percentile(threshV, 97.5))

    return collections.namedtuple("ret", "thresh threshY fitP countDf ci95")(
        truefit.thresh, truefit.threshY, truefit.fitP, truefit.countDf, ci95
    )


def _single_fit(trialDf, do50PctThresh=True, intensName="intensVect"):
    """Fit params of function -- input from a pandas dataframe.
   Args:
       trialDf: dataframe input.
        - has len nTrials not n unique intensities.  
        - must have a column of intensities (contrast, power) with name in param intensName ('intensVect' is
          generic intensity column added by other code above
        - must have isCorr, isFail columns
       do50PctThresh: bool.  If False, use pct=0.6321 for thresh (weibull CDF value at x=p[0])
           default True

   Returns:
        namedtuple: (thresh,threshY,p,cDf,invalidFit)
            invalidFit is a boolean, if too few trials, will be False, cDf will be valid but remaining
            params will be nan

   Notes:
        if index is not unique, the fitting will throw an exception
        Now forces bottom asymptote (4th param to weibull) to zero, needs fixing
    """

    # construct grouping by laser power, compute fracCorr for groups
    cDf = trialDf.groupby(intensName)
    assert len(cDf) > 0, "never should try fit on zero trials"

    invalidFit = False
    # if less than 10% trials in this block, skip fit
    if len(trialDf) < 4 * len(cDf):  # fewer than four trials per level
        if len(trialDf) > 0:  # no need to print message for zero trials
            print("Few trs (%d) in block found, skipping fit" % (len(trialDf)))
        invalidFit = True
    elif len(cDf) < 3:
        print("Too few points/intens levels (%d) in block found, skipping fit" % len(cDf))
        invalidFit = True
        # breakpoint()

    # compute nCorr, nFail for each
    cDf = cDf[["isCorr", "isFail"]].agg("sum").rename(columns={"isCorr": "nCorr", "isFail": "nFail"})
    cDf = cDf.assign(fracCorr=cDf.nCorr / (cDf.nCorr + cDf.nFail))
    cDf = cDf.assign(**{intensName: cDf.index})
    # remove any nulls
    cDf = cDf.dropna(subset=[intensName, "fracCorr"], axis=0)

    # do fit
    if invalidFit:
        thresh, threshY, p, cDf = nan, nan, [nan, nan, nan], cDf
    else:
        runFn = lambda p, xs, ys: ys - weibullF(p, xs)
        x0 = [np.mean(cDf[intensName]), 1, 0.95]

        ret = scipy.optimize.least_squares(
            runFn, x0, args=(cDf[intensName], cDf.fracCorr), bounds=((0, 0, 0), (100, 100, 1.0))
        )
        p = ret.x

        (thresh, threshY) = weibull_compute_thresh_from_p(p, do50PctThresh=do50PctThresh)

    return collections.namedtuple("ret", "thresh threshY fitP countDf invalidFit")(
        thresh, threshY, p, cDf, invalidFit
    )


#######################
## fits from excel files


class ManyFitsFromParams:

    indexCols = ["Subject", "DateStr", "ExtraChar"]  # const, should not change

    def __init__(self, paramXlsName, resultXlsName, localDataDir):
        """
        Args:
            paramXlsName:
            resultXlsName:
            localDataDir:

        Notes:
            param xls file has columns:
                - Subject
                - DateStr
                - ExtraChar
                - StimType: laserA (alias: laser), laserB, visual [default]
                - MultIntensity:
                - TrialsToDo1Origin: string specifying which trials to use. first trial is 1, not 0.
        """
        self.paramXlsName = paramXlsName
        self.resultXlsName = resultXlsName
        self.localDataDir = localDataDir
        self.resultOutputDir = resultXlsName.replace(".xls", "")

        # set up output dir for mat files
        os.makedirs(self.resultOutputDir, exist_ok=True)

        # read including nans, then convert nans in ExtraChar only to empty str,
        # for better comparisons later
        # also store on disk without multi-index, and use set_index when needed
        self.paramDf = pd.read_excel(paramXlsName, index_col=None, converters={"DateStr": str})
        colsReq = ["Subject", "DateStr", "ExtraChar", "StimType", "MultIntensity", "TrialsToDo1Origin"]
        if not np.all(np.isin(colsReq, self.paramDf.columns)):
            raise RuntimeError("Not all columns are found in params file.  Req are %s" % colsReq)

        # fill in missing values with defaults
        self.paramDf.replace({"ExtraChar": {np.nan: ""}}, inplace=True)
        self.paramDf.replace({"MultIntensity": {np.nan: 1}}, inplace=True)
        if "FitLaserAOrB" in self.paramDf.columns:
            warnings.warn(
                "Params file has FitLaserAOrB column, will be ignored. Use laserA/laserB in StimType"
            )
        self.paramDf.replace({"TrialsToDo1Origin": {np.nan: "all"}}, inplace=True)

        # same for resultDf: read, replace
        if os.path.isfile(resultXlsName):
            self.resultDf = pd.read_excel(resultXlsName, index_col=None, converters={"DateStr": str})
            self.resultDf.replace({"ExtraChar": {np.nan: ""}}, inplace=True)
        else:
            self.resultDf = self._init_blank_resultdf(resultXlsName)

        # set up indexes
        self.paramDf = self._set_df_index(self.paramDf)
        self.resultDf = self._set_df_index(self.resultDf)

        # join index for working with in memory
        self.joinedDf = self.paramDf.join(self.resultDf, how="left")

    def _add_intens_column(self, df, stimType):
        """Figures out the variable name to use for stim levels, processes the correct column,
        then returns df with a new 'intensVect' column
        Should include logic for HADC8, HADC19

        Args:
            df: the behavior mwk/mat df with one row for each trial
        """
        whichMwel = None

        stimType = stimType.lower()
        if stimType == "laser" or stimType == "lasera":
            v0 = df.loc[:, "tLaserPowerMw"]
        elif stimType == "laserb":
            v0 = df.loc[:, "tLaserBPowerMw"]
        elif stimType == "visual" or stimType == "":
            if "tStimContrast" in df.columns:
                whichMwel = "HADC19"
                v0 = np.abs(df.loc[:, "tStimContrast"] - df.loc[:, "tBaseStimContrast"])
            elif "tGratingContrast" in df.columns:
                whichMwel = "HADC8"
                v0 = np.abs(df.loc[:, "tGratingContrast"] - df.loc[:, "tBaseGratingContrast"])
        else:
            raise RuntimeError("Invalid stimType %s (may need to be added)" % stimType)

        # round intensities to 4 sig figs to prevent group splitting by tiny differences
        nSigFig = 3
        v0 = ptMH.math.chop(v0, nSigFig)

        df = df.assign(intensVect=v0)

        return df

    def _init_blank_resultdf(self, xlsname):
        df = pd.DataFrame(
            data=[(nan, "", nan, nan)], columns=("Subject", "DateStr", "ExtraChar", "DoneTimestamp")
        )
        print(df)
        print(type(df))
        return df

    def _set_df_index(self, df):
        """Utility function to set up the index in the two dataframes.  
            Use verify_integrity=True so we catch duplicate index entries dead in their tracks"""
        return df.set_index(self.indexCols, verify_integrity=True)

    def _extraout_name(self, subj=None, dateStr=None, extraChar=None, index=None):
        if index is not None:
            (subj, dateStr, extraChar) = index
        return "fitdata-%04d-%6s%s.h5" % (subj, dateStr, extraChar)

    def _save_extraout(self, index, countDfL):
        """Save extra output data to a separate file
        Args:
            countDfL: vector len 2, each element countDf dataframe for one of two blocks
        Returns:
            outName (string): full path and name of written file"""
        outName = (Path(self.resultOutputDir) / self._extraout_name(index=index)).resolve()
        if os.path.exists(outName):
            os.remove(outName)
        ptMH.dataio.saveDataAsH5(outName, {"countDfByBlockL": countDfL})
        return outName

    def load_extraout(self, index):
        outName = os.path.join(self.resultOutputDir, self._extraout_name(index=index))
        ret = ptMH.dataio.readDataFromH5(outName)
        return ret

    def do_all_fits(self, nBootstrapReps=1000, redoAll=False):
        """Iterate over all rows in params and do fits for all those without valid values in results.

        Args:
            nBootstrapReps: int

        Notes:
            Logic to find the right intensity name is here
        """

        for tR in self.paramDf.itertuples():
            try:
                self.resultDf.loc[tR.Index, :]
            except KeyError:
                # not found in index, insert a new row
                self.resultDf.loc[tR.Index, "DoneTimestamp"] = nan
            if tR.TrialsToDo1Origin == "none":
                print('Skipping (TrialsToDo1O == "none"): %d %s %s' % tR.Index)
                continue

            if redoAll or pd.isnull(self.resultDf.loc[tR.Index, "DoneTimestamp"]):
                # read file
                fName = os.path.join(self.localDataDir, mat_io.file_name_data(tR.Index[0], tR.Index[1]))
                if not Path(fName).exists():
                    log.info("Skipping fit, mat file not found. %s, %s" % (tR.Index, fName))
                    continue
                mb = mat_io.matBehavFile(fName)
                nTrs = len(mb.df)

                mb.df = self._add_intens_column(mb.df, tR.StimType)

                # do fit for each block
                countDfL = [None] * 2

                if tR.TrialsToDo1Origin == "all":
                    desTrIx = np.ones((nTrs,), dtype="b1")
                else:
                    desNs = eval("np.r_[%s]-1" % (tR.TrialsToDo1Origin))
                    desTrIx = np.zeros((nTrs,), dtype="b1")
                    desTrIx[desNs] = True

                print("Starting: %d %s %s" % tR.Index)

                for (iB2N, tB2N) in enumerate([1, 2]):
                    tIx = desTrIx
                    # next line looks like it is intended to select out range for current block - in my example
                    # tBlock2TrialNumber is all nan so this isn't working
                    tIx = tIx & (mb.df.tBlock2TrialNumber == iB2N)
                    if iB2N == 0 and np.sum(tIx) == 0:
                        # may want to select out only block1 from range - not sure if there may still be a bug for some
                        # cases
                        tIx = desTrIx  # not using block2, set true to all in precomputed range

                    if np.sum(tIx) == 0:  # no trials
                        continue  # leave all nans
                    ret = fit_and_bootstrap(
                        mb.df.loc[tIx, :], nBootstrapReps=nBootstrapReps, intensName="intensVect"
                    )

                    # drop an informative error message if we think the param intensity is set wrong
                    nFound = 0
                    for x in ["doLaserStim", "doLaserBStim", "doVisualStim", "doAuditoryStim"]:
                        if x in mb.constDL[-1]:  # use constants at end of session
                            nFound += mb.constDL[-1][x]
                    if np.isnan(ret.thresh) and nFound > 1:
                        warnings.warn("Multiple stim types; fit failed w/ few points: wrong intensity type?")

                    # load back into dataframe
                    addD = collections.OrderedDict()
                    addD["Threshold%d" % tB2N] = ret.thresh
                    addD["ThreshY%d" % tB2N] = ret.threshY
                    addD["CI95Low%d" % tB2N] = ret.ci95[0]
                    addD["CI95High%d" % tB2N] = ret.ci95[1]
                    addD["NBootstrapReps"] = nBootstrapReps
                    addD["P1_B%d" % tB2N] = ret.fitP[0]
                    addD["P2_B%d" % tB2N] = ret.fitP[1]
                    addD["P3_B%d" % tB2N] = ret.fitP[2]

                    self.resultDf = df_makecols(self.resultDf, addD.keys())  # make sure all columns exist
                    warnings.filterwarnings(
                        "ignore", "indexing past lexsort depth may impact performance"
                    )  # avoid catch_warnings context manager: is not thread safe and makes for ugly extra indents
                    self.resultDf.loc[tR.Index, addD.keys()] = pd.Series(addD)  # add data to cells we have

                    countDfL[iB2N] = ret.countDf

                # if only one field for both blocks, put here

                self.resultDf = df_makecols(self.resultDf, ["PdfFile", "ExtraOutFile"])
                toStrL = ["PdfFile", "ExtraOutFile", "DoneTimestamp"]
                self.resultDf = self.resultDf.astype({ x: 'str' for x in toStrL })
                self.resultDf.loc[tR.Index, "PdfFile"] = ""
                # write output file with misc data
                extraOutName = self._save_extraout(tR.Index, countDfL)
                self.resultDf.loc[tR.Index, "ExtraOutFile"] = extraOutName

                # mark done
                self.resultDf.loc[tR.Index, "DoneTimestamp"] = time.strftime("%c")  # req meth for strings

                # save xls after each fit
                self._save_result_xls()

        print("Done.  ResultDf is at %s" % self.resultXlsName)

    def do_all_plots(self, pdfOutputDir, redoAll=False):
        """Generate missing plots.
        Args:
            pdfOutputDir: string path to directory where fit plots go.  This is not a parameter to the class because 
            only this function uses it.  Class can be used to do fits or to read fit data, neither needs this param.
        """
        os.makedirs(pdfOutputDir, exist_ok=True)
        sns.set_style("darkgrid")  # probably should push and pop this

        for (tIndex, tR) in self.joinedDf.iterrows():
            subj, datestr, extrachar = tIndex
            if tR.TrialsToDo1Origin == "none":
                print('Skipping (TrialsToDo1O == "none"): %d %s %s' % tIndex)
                continue

            if redoAll or (tR.PdfFile == ""):

                pdfname = os.path.join(pdfOutputDir, "%04d-%s%s.pdf" % (subj, datestr, extrachar))
                figH = self.fig_psych_curves(tIndex)
                figH.savefig(pdfname, format="pdf")
                figH.set_visible(False)
                plt.close(figH)
                self.resultDf.loc[tIndex, "PdfFile"] = pdfname  # note we set resultDf, but read joinedDf
                print("do_all_plots: Done with pdf: %d %s %s" % tIndex)

        # save to disk
        self._save_result_xls()
        print("do_all_plots: Done.  Saved resultDf at %s" % self.resultXlsName)

    def _save_result_xls(self, skipJoin=False):
        """Save output, renaming old results file out of the way first.
        We never write the params df.

        Anything that changes resultDf should call this.

        When done, redo join into joinedDf, unless skipJoin=True
        """
        self.resultDf = self.resultDf.reset_index()  # needed before excel
        self.resultDf.dropna(subset=["DoneTimestamp"], inplace=True)
        if os.path.exists(self.resultXlsName):
            os.rename(self.resultXlsName, self.resultXlsName.replace(".xls", "-backup.xls"))

        self.resultDf.to_excel(self.resultXlsName)
        self.resultDf = self._set_df_index(self.resultDf)  # restore index

        self.joinedDf = self.paramDf.join(self.resultDf, how="left")

    def fig_psych_curves(self, index, xMinMax=None):
        """Create a psychfun figure, with both block2s if present in the file
        - Log x axis
        - returns all handles, you can delete some if necessary
        - uses 100 pts in plot
        - y axis is percent correct  (0...100)

        Args:
            index: tuple: (subj,dateStr,extraChar)
        Returns:
            figH: created figure handle
        """
        xout = self.load_extraout(index)

        nBlocks = np.sum([x is not None for x in xout["countDfByBlockL"]])

        figH = plt.figure(figsize=6 * a_((1, 2)) * a_((1, 0.75)))
        axHL = [None] * 2
        for iB in range(nBlocks):
            axHL[iB] = plt.subplot(2, 1, iB + 1)
            self._plot_oneblock_psych_curve(index, iB + 1, xMinMax=xMinMax, xout=xout)
            # if index[0] == 1167:
            #    print(xout['countDfByBlockL'][iB]) # debug

            if nBlocks == 2:
                if iB == 0:
                    plt.xlabel("")
                elif iB == 1:
                    xL0 = axHL[0].get_xlim()
                    xL1 = axHL[1].get_xlim()
                    outerRange = (np.min((xL0[0], xL1[0])), np.max((xL0[1], xL1[1])))
                    axHL[0].set_xlim(outerRange)
                    axHL[1].set_xlim(outerRange)
        return figH

    def _plot_oneblock_psych_curve(self, index, block2N=1, xMinMax=None, xout=None):
        """Plot curve into an axes for one block

        Args:
            index: tuple: (subj,dateStr,extraChar)
            block2N: 1-origin block2 number (i.e. in [1, 2], not in [0, 1]
            xout: extra out data file, must be read elsewhere
        """
        ax = plt.gca()

        tR = self.joinedDf.loc[index, :]

        countDf = xout["countDfByBlockL"][block2N - 1]
        if countDf is None:
            raise RuntimeError("block2N %d not found in result data" % block2N)

        if xMinMax is None:
            xMinMax = (countDf.index.min() * 0.5, countDf.index.max() * 1.5)

        # for clarity below: extract B1/B2 values into a namedtuple
        bN = block2N
        tT = collections.namedtuple("_", "P CI95 Threshold ThreshY")(
            (tR["P1_B" + str(bN)], tR["P2_B" + str(bN)], tR["P3_B" + str(bN)]),
            (tR["CI95Low" + str(bN)], tR["CI95High" + str(bN)]),
            tR["Threshold" + str(bN)],
            tR["ThreshY" + str(bN)],
        )

        xs = 10 ** np.linspace(np.log10(xMinMax[0]), np.log10(xMinMax[1]))
        ys = weibullF(tT.P, xs)
        lH = plt.plot(xs, ys * 100, "k")

        plt.plot(countDf.index, countDf.fracCorr * 100, "b.")
        plt.plot(tT.CI95, tT.ThreshY * 100 * a_([1, 1]), color="0.4")
        plt.plot(tT.Threshold * a_((1, 1)), [0, tT.ThreshY * 100], "--", color="0.4", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylim([0, 100])

        ax.xaxis.set_major_locator(plt.LogLocator(subs="all"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "%2g" % (x)))
        plt.xlabel(tR.StimType)
        plt.ylabel("% correct")
        plt.title(
            "%d %s %s: thresh %.3g [%.3g-%.3g]" % tuple([i for i in index] + [tT.Threshold] + list(tT.CI95))
        )

        plt.tight_layout()
