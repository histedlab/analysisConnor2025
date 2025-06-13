"""caiman_1p_processing_utils.py:
201020 MH: we think this is mainly code for regressing out 1p stim artifacts with Caiman.
May just remove this at some point.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
import os
import sys
import logging
import scipy.ndimage as ndimage
import scipy.sparse
import scipy.stats as ss
import IPython, ipywidgets

sns.set_style("darkgrid")

# our modules
import mworksbehavior as mwb
import pytoolsMH as ptMH
from .. import mwkfiles
from . import io

a_ = np.array
r_ = np.r_
nax = np.newaxis

logger = logging.getLogger(__name__)

## Util consts

clist = [(0, 0, 0), (0, 1, 0)]
cmapGreen = mpl.colors.LinearSegmentedColormap.from_list("G", clist)

## Util fns


def reshape_vec_by_stims(tV, mwf):
    startNs = get_startns(mwf)
    nLevels = len(startNs)
    aL = []
    for iL in range(nLevels):
        a0 = a_([tV[x - mwf.constS.counterNPreTrig : x + mwf.constS.counterNPostTrig] for x in startNs[iL]])
        a1 = np.vstack(a0)
        aL.append(a1)
    a2 = np.stack(aL, axis=0)
    return a2


def get_startns(mwf):
    """Return startNs: array giving start frame of each stimulus
    Returns:
        startNs (array vec of array vecs, shape (nStimLevels, nReps)).  But obj array of index arrays.
           So index as startNs[stimN][repN] not startNs[stimN,repN]."""
    levels = np.unique(mwf.stimDf[mwf.levelVar])
    nLevels = len(levels)

    startNL = []
    periodFr = mwf.constS.counterNPreTrig + mwf.constS.counterNPostTrig
    for (iL, tL) in enumerate(levels):
        stimNs = a_((mwf.stimDf[mwf.levelVar].values == tL).nonzero()[0], dtype="int")
        startNs = stimNs * periodFr + mwf.constS.counterNPreTrig
        # startNs = np.concatenate((startNs, np.zeros(iL,)))  # for testing only, dbeug
        startNL.append(startNs)
    return np.asarray(startNL, dtype="O")


def reshape_by_stims(mwf, arr, axis=-1):
    return np.apply_along_axis(reshape_vec_by_stims, axis, arr, mwf)


## A class to hold reshaped and averaged cnm data


class CnmAnalysis:
    """If we switch to reading mwf internally we need to drop artifcat frames appropriately.



    Notes:
        - Put any computation code that isn't TOO slow in init.  Can do longer or larger computation
            in other methods to defer those steps
        - Dims of all cnmByStim: nCells, nStims, nReps, nAvgTp
        - When we add code to read the mwk file, remove the artifact frames on read so later code doesn't need to know about them

    """

    def __init__(self, rootdir, useStimRange="all", nStimArtifactFrs=1, subdir=None,
                 dd2=None, stackfile='imageFrames-mc.tif', stackout_subdir=None):
        if dd2 is None:
            self.dd2 = mwb.imaging.consts.DataDir2p(rootdir, subdir=subdir, stackout_subdir=stackout_subdir)
        else:
            self.dd2 = dd2
        self.cnm = self.dd2.get_caiman_results()
        self.mwf = self._load_mwk_file(self.dd2, useStimRange, nStimArtifactFrs)
        self.nStimArtifactFrs = nStimArtifactFrs

        (self.nCells, self.nKeptFrames) = self.cnm.S.shape

        # convert A to dense ndarray in memory
        if hasattr(self.cnm.A, "todense"):  # is a matrix or sparse
            # is sparse, convert to ndarray, else assume it is already
            self.cnm.A = self.cnm.A.todense().A  # A attrib returns ndarraay, don't confuse As

        # back compat
        if not hasattr(self.cnm, "frPerS"):
            logger.warning("Caiman output missing frPerS (added 180622), assuming 5")
            self.cnm.frPerS = 5

        # now reshape by stims
        self.cnmByStim = argparse.Namespace()
        for tK in vars(self.cnm):
            tV = getattr(self.cnm, tK)
            if hasattr(tV, "ndim") and tV.ndim > 1:
                desN = np.nonzero(a_(tV.shape) == self.nKeptFrames)[0]
                if len(desN) == 1:
                    reshaped = reshape_by_stims(self.mwf, tV, axis=desN)
                    setattr(self.cnmByStim, tK, reshaped)
                    logger.debug("cnmByStim %s shape old %s new %s" % (tK, tV.shape, reshaped.shape))
                    continue
                logger.debug("skipping field %s, shape %s" % (tK, tV.shape))
        cbs = self.cnmByStim  # shorthand
        (cbs.nCells, cbs.nStims, cbs.nReps, cbs.nAvgTp) = self.cnmByStim.YrA.shape
        self.cnmByStim.xs = (
            r_[0 : cbs.nAvgTp] - self.mwf.constS.counterNPreTrig
        )  # artifact frs presumed removed

        # compute selected zscores - taken over all reps and
        arr = self.cnmByStim.S
        self.cnmZScore = argparse.Namespace()
        baseNs = r_[0 : self.mwf.constS["counterNPreTrig"]]
        self.cnmByStim.baseNs = baseNs
        baseStd = np.std(arr[:, :, :, baseNs], axis=(1, 2, 3))
        baseMean = np.mean(arr[:, :, :, baseNs], axis=(1, 2, 3))

        baseStdT = baseStd.copy()
        lowThresh = np.percentile(baseStd, 10)
        baseStdT[baseStd < lowThresh] = np.max((lowThresh, np.min(baseStdT[baseStdT > 0])))

        self.cnmZScore.S = (self.cnmByStim.S - baseMean[:, nax, nax, nax]) / baseStdT[:, nax, nax, nax]

        # compute S normalized by var over mean
        #  See 180516-1p-stim-analysis-v2/develop/180621-MH-backgroundnorm.ipynb
        baseVar = baseStd ** 2
        nSamps = np.prod(self.cnmByStim.S.shape[1:])
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # there will be divide by zero; nans will propagate properly
            normC = baseVar / baseMean
            minThresh = np.min(
                (np.percentile(normC[~np.isnan(normC)], 10), nSamps / 10)
            )  # thresh is 10th percentile or only resp in 10 bins, whichever is smaller
            normC[normC < minThresh] = minThresh
        self._S_norm_factor = normC
        self.cnmByStim.SNormCt = self.cnmByStim.S / normC[:, nax, nax, nax]

        # compute f0 image.  must read the orig/MC'd stack to get this.  We don't know baseNs until after
        # the code above is run, so we can't do this in the caiman run code/pipeline.  Doesn't take too long.
        im = io.load_tiff(os.path.join(rootdir, stackfile))
        self.cnmByStim.f0_im = im[self.cnmByStim.baseNs, :, :].mean(axis=0)
        del im  # clear it out of memory (at least at next gc); I guess we could memmap it here, right?

    def _load_mwk_file(self, dd2, useStimRange="all", nStimArtifactFrs=1):
        """Returns:
            mwf
        """

        mwf = mwkfiles.ChRMap1MWKFile(dd2.h5name, firstTrStartTimeUs=None, stims_to_keep=useStimRange)
        #                                    discardAfterTimeUs=500000000)
        mwf.save_stim_params(dd2.h5stimsname)
        mwf.compute_imaging_constants()
        mwf.drop_artifact_frames(nStimArtifactFrs)

        return mwf


class CnmPlotter:
    def __init__(self, cnmAnalysis):
        """
        Notes:
            former param outdir now hardcoded to rootdir/figures
        """
        assert hasattr(cnmAnalysis, "cnmByStim"), "pass in a CnmAnalysis instance"
        self.ca = cnmAnalysis
        self.outdir = os.path.join(self.ca.dd2.rootdir, "figures")
        os.makedirs(self.outdir, exist_ok=True)

    def plot_mean_tc_manymeasures(self, doSave=False, plot_dff=False):
        fig = plt.figure(figsize=r_[1, 0.75] * 12 * r_[0.75, 1])
        gs = mpl.gridspec.GridSpec(3, 3)
        ca = self.ca  # shortcut

        xs = ca.cnmByStim.xs
        ax1 = plt.subplot(gs[0, 0])
        ys = ca.cnmByStim.S.mean(axis=(0, 2))  # mean over cells, reps
        plt.plot(xs, ys.T)
        plt.title("S")
        leg = plt.legend(["%.3gmW" % d for d in ca.mwf.levelDf.tAPeakPowerMw])
        plt.ylabel("a.u.  ($a\cdot S$)")

        ax3 = plt.subplot(gs[0, 1])
        plt.plot(xs, ca.cnmZScore.S.mean(axis=(0, 2)).T)
        plt.title("S zscore average")
        plt.ylabel("zscore")

        # plot new measure
        ax = plt.subplot(gs[0, 2])
        ys = np.nanmean(ca.cnmByStim.SNormCt, axis=(0, 2)).T
        Fs = 1 / 5.0
        plt.plot(xs, ys / Fs)
        plt.title("S norm (var/mean)")
        plt.ylabel("approx rate")
        plt.xlabel("frames")

        # plot C
        ax2 = plt.subplot(gs[1, 0])
        plt.plot(xs, ca.cnmByStim.C.mean(axis=(0, 2)).T)
        plt.title("C")

        if plot_dff is True:
            ax = plt.subplot(gs[1, 1])
            ys = ca.cnmByStim.F_dff.mean(axis=(0, 2)).T
            plt.plot(xs, ys)
            plt.title("F_dff")

        # plot C and background separately
        nBasePts = 5
        nPre = ca.mwf.constS.counterNPreTrig
        ax = plt.subplot(gs[1, 2])
        YrA = ca.cnmByStim.YrA[:, :, :, :].mean(axis=2)
        C = ca.cnmByStim.C[:, :, :, :].mean(axis=2)
        f_tc = ca.cnmByStim.f[:, :, :, :].mean(axis=2)
        b_weight_all_cells = np.dot(ca.cnm.b.T, ca.cnm.A)
        f_all = np.dot(f_tc.T, b_weight_all_cells).T
        y1 = np.mean(f_all, axis=0)
        y1 = y1 - np.mean(y1[:, :nPre])
        pH1 = plt.plot(xs, y1.T, "-", label="backg sum", alpha=0.5)
        y2 = np.mean(C, axis=0)
        y2 = y2 - np.mean(y2[:, :nPre])
        pH2 = plt.plot(xs, y2.T, label="C")
        for iC in range(len(pH1)):
            pH2[iC].set_color(pH1[iC].get_color())
        plt.title("dim: bg sum; solid: C; all mean sub")

        basem = ca.cnmByStim.f[:, :, :, ca.cnmByStim.baseNs].mean(axis=(1, 2, 3))  # res is shape (nComp,)
        nBack = ca.cnm.f.shape[0]
        for iComp in range(np.min((2, nBack))):
            ax = plt.subplot(gs[2, iComp])
            ys = ca.cnmByStim.f[iComp, :, :, :].mean(axis=(1)).squeeze() - basem[iComp]
            pH = plt.plot(xs, ys.T)
            ax.set_title("backg comp %d, mean sub" % iComp)
            ax.set_xlabel("frames")

        for ax in [ax1, ax2]:
            ax.axvline(0, color="k", lw=0.25)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle("%s: averages across %d cells" % (ca.mwf.basename, ca.cnmByStim.nCells))
        if doSave:
            fig.savefig(os.path.join(self.outdir, "fig-cell-avgs.pdf"))

    def plot_cell_tc_individually(self, kind):
        """kind: 'SZscore', 'Y', 'C' """

        ca = self.ca  # shorthand
        cbs = self.ca.cnmByStim

        fig = plt.figure(figsize=r_[1, 0.75] * 25)
        nP = int(np.ceil(np.sqrt(ca.nCells)))
        gs = mpl.gridspec.GridSpec(nP, nP)
        axL = [None] * ca.nCells
        xs = ca.cnmByStim.xs

        if kind == "SZscore":
            stimArr = ca.cnmZScore.S.mean(axis=2)
            extra_text = "S Zscore"
            minY = 3
        elif kind == "SNorm":
            stimArr = ca.cnmByStim.SNormCt.mean(axis=2) * ca.cnm.frPerS
            extra_text = r"$\approx$ rate"
            minY = 3
        elif kind == "C":
            stimArr = ca.cnmByStim.C.mean(axis=2)
            extra_text = "C"
            minY = 3
        elif kind == "F_dff":
            stimArr = a_(ca.cnmByStim.F_dff.mean(axis=2), dtype="f8")
            extra_text = "F_dff"
            minY = 1
        elif kind == "YrA":
            stimArr = ca.cnmByStim.YrA.mean(axis=2)
            extra_text = "fluo - YrA"
            minY = 1
        else:
            raise RuntimeError("unknown plot kind %s" % kind)
        for iC in range(ca.nCells):
            ax = plt.subplot(gs[iC])
            axL[iC] = ax

            ys = stimArr[iC, :, :].T
            if ~np.all(np.isnan(ys)):
                plt.plot(xs, ys)
                yL = (np.min(ys) - np.ptp(ys) * 0.2, np.max((minY, ptMH.math.chop(np.max(ys), 1))))
                ax.set_ylim(yL)

            plt.tick_params(labelbottom=False, labelleft=True)
            ax.axvline(0, color="k", lw=0.25)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=1, integer=True, min_n_ticks=2))
            # ax.set_yticks(np.concatenate(((0,),yL[1:])))
            plt.annotate(
                iC + 1,  # start at 1, not zero
                xy=(0, 1.0),
                xycoords="axes fraction",
                fontsize=8,
                xytext=(2, -2),
                textcoords="offset points",
                ha="left",
                va="top",
            )
            # if iC > 15:  # debug
            #    break
            if iC % nP == 0:
                plt.ylabel(extra_text)

        sns.despine()

        ext_cln = re.sub("[^\S]*", "", kind).strip().lower()
        plt.suptitle("%d cells - %s" % (ca.nCells, extra_text))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(self.outdir, "fig-cells-%s.pdf" % ext_cln))
        plt.draw()

    def plot_background_components_dff(self, doSave=True):
        """doSave: whether to write output PDF"""
        ca = self.ca
        f0_im = ca.cnmByStim.f0_im
        nBack = ca.cnmByStim.f.shape[0]

        B = ca.cnm.b.reshape([ca.cnm.d1, ca.cnm.d2, nBack], order="F")
        B = np.moveaxis(B, -1, 0)

        nBack = B.shape[0]
        nPan = int(np.ceil(np.sqrt(nBack + 2)))
        fig = plt.figure(figsize=r_[1, 0.75] * nPan * 4)
        gs = mpl.gridspec.GridSpec(nPan, nPan)

        def _symm_max_r(im, alpha=0):
            """Util fn to get symmetric axis limits based on max in either direction, with percentile option"""
            r0 = np.max(np.abs((np.percentile(im, alpha), np.percentile(im, 100 - alpha))))
            return r0 * r_[-1, 1]

        BNorm = a_([B[iB, :, :] / B[iB, :, :].max() for iB in range(B.shape[0])])
        fNorm = f0_im / f0_im.max()
        fn_low = fNorm
        fn_low[fn_low < 0.05] = 0.05
        for iB in range(nBack):
            ax = plt.subplot(gs[iB])
            Bdff = (BNorm[iB, :, :] - fNorm) / fn_low
            plt.imshow(Bdff, cmap="RdBu_r", clim=_symm_max_r(Bdff, alpha=1))
            plt.title(r"B[%d]  $(\Delta B/F)$" % iB)
            cb = plt.colorbar()
            cb.set_label("$\\frac{B_{norm}[%d]-f0_{norm}}{f0_{norm}}$" % iB)

        # diff
        if nBack > 1:
            ax = plt.subplot(gs[iB + 1])
            plt.title("B[1]-B[0] (normed): ($\Delta B$)")
            Bd = BNorm[1, :, :] - BNorm[0, :, :]
            # r0 = np.max(np.abs((np.percentile(Bd,1),np.percentile(Bd,99))))
            plt.imshow(Bd, clim=_symm_max_r(Bd, alpha=2), cmap="RdBu_r")
            cb = plt.colorbar()
            cb.set_label("$B_{norm}[1]-B_{norm}[0]$")

        # f0 norm
        ax = plt.subplot(gs[iB + 2])
        plt.title("$f0_{norm}$")
        plt.imshow(fNorm, cmap="gray", clim=[0, np.percentile(fNorm, 98)])
        plt.colorbar()

        plt.suptitle("background components, dff: relative to **f0**, not baseline backg image")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if doSave:
            fig.savefig(os.path.join(self.outdir, "fig-backg-comp.pdf"))
        plt.draw()

    def plot_backg_timecourse_diff(self, doSave=True):
        """we plot dff, approximating spiking, at least for low frame rates"""

        nBack = self.ca.cnm.b.shape[1]
        nRows = np.max((2,nBack))

        nCols = np.min((2, np.max((2,nBack))))
        fig = plt.figure(figsize=r_[1, 0.75] * r_[nCols, nRows] * 4)
        gs = mpl.gridspec.GridSpec(nRows, nCols)
        xs = r_[0 : self.ca.cnmByStim.nAvgTp] - 25  # frames

        axDerivs = plt.subplot(gs[0, 1])
        for iB in range(nBack):
            ax = plt.subplot(gs[iB, 0])
            fm = self.ca.cnmByStim.f[iB, :].mean(axis=1).T
            plt.plot(xs, fm)
            plt.title("comp %d" % iB)
            axDerivs.plot(xs[1:], np.diff(fm, axis=0), ".-")
            if iB == nBack - 1:
                ax.set_xlabel("frames")

        axDerivs.set_title("derivs")
        axDerivs.legend(["comp %d" % id for id in range(nBack)])

        ax = plt.subplot(gs[1, 1])
        plt.title("deriv sum")
        fm_diff = np.diff(self.ca.cnmByStim.f.mean(axis=2), axis=2)
        plt.plot(xs[1:], fm_diff.sum(axis=0).T)
        ax.set_xlabel("frames")

        plt.suptitle("backg timecourses, derivatives, and sum deriv")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if doSave:
            fig.savefig(os.path.join(self.outdir, "fig-backg-tc-derivs.pdf"))
        plt.draw()

    def plot_frame_means(self, doSave=True):
        ca = self.ca
        mwf = ca.mwf
        dfof = ca.dd2.get_output_stack("dfof.tif")  # loads dfof from disk

        fig = plt.figure(figsize=r_[1, 0.75] * [2, 2] * 5)
        gs = mpl.gridspec.GridSpec(2, 2)

        r0 = np.mean(dfof, axis=(1, 2))
        nTp = len(r0) // mwf.nstim
        xv = (r_[0:nTp] - mwf.constS.counterNPreTrig) / ca.cnm.frPerS
        r1 = np.reshape(r0, [mwf.nstim, nTp])
        nBaseSubPts = 5
        nPostAvgPts = 2
        mean0 = np.mean(r1[:, mwf.constS.counterNPreTrig - nBaseSubPts : mwf.constS.counterNPreTrig], axis=1)

        # first plot
        ax = plt.subplot(gs[0, 0])
        pH = plt.plot(xv, r1.T - mean0, ".-")
        for (iX, x) in enumerate(mwf.levelDf.tAPeakPowerMw):
            pH[iX].set_label("%.3g mW" % x)

        plt.plot(r_[-nBaseSubPts, 0] / ca.cnm.frPerS, r_[1, 1] * -0.5, "k", label="base sub")
        plt.plot(r_[1, nPostAvgPts] / ca.cnm.frPerS, r_[1, 1] * -0.5, "r", marker=".", label="resp mean")
        plt.legend(frameon=False)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("avg $\Delta F/F$")

        # plot 2
        ax = plt.subplot(gs[0, 1])
        plt.plot(mwf.levelDf.tAPeakPowerMw, r1[:, mwf.constS.counterNPreTrig + nPostAvgPts] - mean0, ".-")
        ax.set_xscale("log")
        # ax.set_xlim([0.001, 2.0])
        ax.set_xticklabels(["%.3g" % x for x in ax.get_xticks()])
        ax.set_ylabel("avg $\Delta F/F$")
        ax.set_xlabel("power (mW)")

        # fig fixups
        fig.suptitle("mean response over entire imaging frame")

        if doSave:
            fig.savefig(os.path.join(self.outdir, "fig-frame-tcs.pdf"))

    def plot_cell_mean_responses(self, plotType=None, doSave=True):
        """
        Args:
          plotType: 'cellSMeans', 'reconDfof'
        """
        fig = plt.figure(figsize=r_[1, 0.75] * [2, 3] * 5)
        gs = mpl.gridspec.GridSpec(3, 2)

        ca = self.ca
        cbs = ca.cnmByStim
        xv = ca.mwf.levelDf.tAPeakPowerMw
        nPre = ca.mwf.constS.counterNPreTrig
        nBase = 5
        doCellSMeans = True

        if plotType == "cellSMeans":
            tMat = cbs.SNormCt.mean(axis=2)
            ylab = "$S_{est}$"
        elif plotType == "reconDfof":
            # do reconstructed means: F in each cell ROI
            YrA = ca.cnmByStim.YrA[:, :, :, :].mean(axis=2)
            C = ca.cnmByStim.C[:, :, :, :].mean(axis=2)
            f_tc = ca.cnmByStim.f[:, :, :, :].mean(axis=2)
            b_weight_all_cells = np.dot(ca.cnm.b.T, ca.cnm.A)
            f_all = np.dot(f_tc.T, b_weight_all_cells).T
            tMat = C + YrA + f_all
            baseMean = np.mean(tMat[:, :, nPre - nBase : nPre], axis=2)
            tMat = (tMat - baseMean[:, :, nax]) / baseMean[:, :, nax]
            ylab = "$\Delta F/F$"
        else:
            raise RuntimeError("Unknown plotType %s" % plotType)

        cellV = np.mean(tMat[:, :, nPre : nPre + 2], axis=2)
        baseV = np.mean(tMat[:, :, nPre - nBase : nPre], axis=2)
        cellSem = ss.sem(cellV, axis=0, nan_policy="omit")
        cellMean = np.nanmean(cellV, axis=0)
        baseMean = np.nanmean(baseV, axis=0)

        def _p():
            ax.plot(xv, cellMean)
            ax.plot(xv, cellMean + cellSem, "0.8")
            ax.plot(xv, cellMean - cellSem, "0.8")

        ax = plt.subplot(gs[0, 0])
        _p()
        ax.set_ylabel(ylab)
        ax.set_xlabel("Power (mW)")

        ax = plt.subplot(gs[0, 1])
        _p()
        ax.set_xscale("log")
        ax.set_xticklabels(["%.3g" % x for x in ax.get_xticks()])
        ax.set_xlabel("Power (mW)")

        # plot distribution at 0.1 and 0.5
        ax = plt.subplot(gs[1, 0])
        for iP in range(5):
            ptMH.plotting.cdfplot(cellV[:, iP] - baseMean[iP], label="%.3g" % xv[iP])
        plt.legend()
        ax.set_xscale("symlog", linthreshx=0.1)
        ax.set_xlim([-0.15, 2.5])
        ax.set_xticklabels(["%.3g" % x for x in ax.get_xticks()])
        ax.set_xlabel(ylab)

        # plot all cells on log scale
        ax = plt.subplot(gs[1, 1])
        plt.plot(xv, cellV.T)
        ax.set_xscale("log")
        ax.set_xticklabels(["%.3g" % x for x in ax.get_xticks()])
        ax.set_xlabel("Power (mW)")

        # fig fixups
        fig.suptitle("Mean cell responses - %s, N=%d" % (ylab, cellV.shape[0]))

        if doSave:
            fig.savefig(os.path.join(self.outdir, "fig-cell-meanresps-%s.pdf" % plotType))


class CellComponentPlotter1p:
    def __init__(self, ca):
        """"""
        # computations
        self.ca = ca
        self.component_outpath = os.path.join(self.ca.dd2.rootdir, "figures-cellcomps")

        # convert A to (nY, nX, nCells) from (nY*nX, nCells) - make sure ndarray too
        self.A = ca.cnm.A.reshape(np.shape(ca.cnm.Cn) + (-1,), order='F')  # localCorr is Cn
        assert not scipy.sparse.issparse(self.A), "ca should make it dense"
        assert type(self.A) == np.ndarray, "no matrices here or bugs on dot() will occur below"

        self.powers = ca.mwf.levelDf.tAPeakPowerMw

    def component_plot(self, cellN):
        ca = self.ca
        nPre = self.ca.mwf.constS.counterNPreTrig
        nPost = self.ca.mwf.constS.counterNPostTrig
        nFrRep = nPre + nPost

        YrA = ca.cnmByStim.YrA[cellN, :, :, :].mean(axis=1)
        SNormCt = ca.cnmByStim.SNormCt[cellN, :, :, :].mean(axis=1)
        C = ca.cnmByStim.C[cellN, :, :, :].mean(axis=1)
        f_tc = ca.cnmByStim.f[:, :, :, :].mean(axis=2)
        b_weight_this_cell = np.dot(ca.cnm.b.T, ca.cnm.A[:, cellN])
        f_all = np.dot(f_tc.T, b_weight_this_cell).T
        xs = r_[0:nFrRep] - nPre - 1

        def panel_fix():
            plt.axvline(0, lw=0.5)

        fr, fc = (3, 3)
        fig = plt.figure(figsize=r_[1, 0.75] * 5 * r_[fc, fr])
        gs = mpl.gridspec.GridSpec(fr, fc)

        ax = plt.subplot(gs[0, 0])
        pH = plt.plot(xs, C.T)
        panel_fix()
        plt.title("C")

        ax = plt.subplot(gs[1, 0])
        for iS in range(ca.cnmByStim.nStims):
            ph = plt.plot(xs, (f_all + C).T[:, iS], label="C+background estimate")[0]
            plt.plot(xs, (f_all + C + YrA).T[:, iS], ":", label="data", color=ph.get_color())
        plt.plot(xs, f_all.T, ".-", markersize=2, color="gray", label="background")
        panel_fix()
        plt.xlabel("frames")
        plt.title("back: gray, back+C; back+C+YrA")

        ax = plt.subplot(gs[0, 1])
        legPH = plt.plot(xs, SNormCt.T)  # will use this for a legend on the image panel below
        plt.title("Spike est, ct norm")
        yl0 = ax.get_ylim()
        if yl0[1] < 0.5:
            ax.set_ylim((yl0[0], 0.5))
        panel_fix()

        # image panels
        def drawcontours(**kwargs):
            tMask = self.A[:, :, cellN]
            topLev = np.max(tMask) * 0.5
            cs = plt.contour(
                self.A[:, :, cellN] / topLev, levels=[0.1, 0.5], colors="w", linewidths=2, **kwargs
            )
            ax = plt.gca()
            ax.grid(False)
            xys = cs.collections[0].get_segments()[0]  # use first/only contour
            mx = np.mean(xys[:, 0], axis=0)
            my = np.mean(xys[:, 1], axis=0)
            # center the axis on the contour.
            axV = r_[-30, 30, -30, 30] + r_[mx, mx, my, my]
            # correct if the axis goes off the edge of the image
            runoverV = r_[np.min((axV[0], 0)), np.max((0, axV[1]-self.ca.cnm.d2)),
                        np.min((axV[2], 0)), np.max((0, axV[3]-self.ca.cnm.d1))]
            offsetX = np.sum(-runoverV[0:2])  # if both zero,off=0.  If one nonzero,
            offsetY = np.sum(-runoverV[3:5])  # off is that. If both nonzero, tiny frame, avg a bit and don't worry
            axV[0:2] = axV[0:2]+offsetX
            axV[2:5] = axV[2:5]+offsetY
            axV[axV<0] = 0  # after shift, if any still negative, set to zero. (should be small)
            plt.axis(axV)

            ax.invert_yaxis()



        ax = plt.subplot(gs[2, 0])
        fov = ca.cnm.Cn
        fov = (fov + np.max(fov / 100)) / ndimage.gaussian_filter(fov, sigma=(10, 10))
        plt.imshow(fov, cmap=cmapGreen)
        drawcontours(alpha=0.5)

        ax = plt.subplot(gs[1, 1])
        plt.imshow(fov, cmap=cmapGreen)
        drawcontours(alpha=0.5)
        ax.set_anchor("W")
        ax.set_xlim([0, fov.shape[1]])  # reset limits so we can see it all
        ax.set_ylim([0, fov.shape[0]])
        ax.invert_yaxis()

        # make a legend on the figure, in empty space
        # legH = fig.legend(pH, ['%gmW'%d for d in self.powers],
        #                  loc='center', fontsize=12, bbox_to_anchor=(0.4,0.4))
        legH = plt.legend(
            pH, ["%gmW" % d for d in self.powers], loc="upper left", fontsize=10, bbox_to_anchor=(1, 1)
        )

        ax = plt.subplot(gs[2, 1])
        b0 = self.A[:, :, cellN]
        plt.imshow(b0)
        plt.colorbar()
        drawcontours()
        ax.set_xlim()
        plt.title("contour levels: 10%, 50%")

        # plot residuals in a separate big figure
        plt.subplot(gs[0:3, 2])
        base0 = np.mean((f_all + C)[0, :])
        off0 = np.ptp((f_all + C + YrA)[-1, :]) * 1.1
        for iS in range(ca.cnmByStim.nStims):
            off = off0 * iS
            ph = plt.plot(xs, (f_all + C).T[:, iS] + off - base0, label="C+background estimate")[0]
            ph2 = plt.plot(
                xs, (f_all + C + YrA).T[:, iS] + off - base0, ":", label="data", color=ph.get_color()
            )
        plt.title("solid: C+background est; dotted: add YrA")
        plt.ylabel("mean subtracted, offset for visual clarity")
        panel_fix()

        fig.suptitle("%s - Cell %d" % (ca.dd2.mwkbasename, cellN))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def write_all_component_plots(self, cellNs="all"):
        ca = self.ca
        if cellNs == "all":
            cellNs = r_[0 : ca.cnmByStim.nCells]

        os.makedirs(self.component_outpath, exist_ok=True)
        progbar = ipywidgets.IntProgress(min=0, max=ca.cnmByStim.nCells)
        proglabel = ipywidgets.Label(value="")
        # IPython.display.display(ipywidgets.HBox([progbar,proglabel]))
        print("nTotal: %d " % len(cellNs))
        for iC in cellNs:
            fig = self.component_plot(iC)
            fig.savefig(os.path.join(self.component_outpath, "cell-%03d" % iC))
            plt.close()

            progbar.value = iC
            proglabel.value = "%d of %d cells done" % (iC + 1, ca.cnmByStim.nCells)
            if iC % 10 == 0:
                print(iC, end=" ")
        print(" Done.")
