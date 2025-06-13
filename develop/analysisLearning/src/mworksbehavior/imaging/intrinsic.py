from __future__ import print_function

"""Analyze intrinsic imaging data.  Also contains code for reading 

Todo:
    - should break out the AnalysisStacks hardcoded path into our process_filenames() function
    - may wish to merge the RMA and RMP objects (RMP may need paths from RMA, etc.)      

171108: v2 (intrinsic_imaging2) - for randomization.
171031: created MH, adapting code from BA/MH
180501: moved to package mworksbehavior (to be renamed) MH
190722: BA: changes for MWK2 files
"""

# imports: anaconda standard
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import os
import warnings
import types
import glob
import subprocess as sp
import packaging
from packaging.version import Version
from argparse import Namespace

# our imports
from .. import mwkfiles
import pytoolsMH as ptMH
import mworksbehavior.mwk_io

a_ = np.asarray
r_ = np.r_
n_ = np.newaxis
nax = np.newaxis


# compat check
if Version(pd.__version__) < Version("0.20"):
    warnings.warn("**Must upgrade pandas version or silent bugs may occur**: version is %s" % pd.__version__)

### util fns
def _normarr(im, axis=-1):
    """Normalize a multi-dim array by taking max/min along one axis.  Return a copy"""
    otherax = tuple(np.delete(range(im.ndim), axis))
    mn = np.min(im, axis=otherax)
    mx = np.max(im, axis=otherax)
    outL = []
    for iA in range(im.shape[axis]):
        im2 = np.take(im, iA, axis=axis)
        outL.append((im2 - im2.min()) / np.ptp(im2))
    return np.stack(outL, axis=axis)


class RetinoMapAnalyzer:
    def __init__(self, rootdir, mwkname=None, redoCached=False):
        """For now, this only processes filenames
        MH 180614"""
        self.rootdir = rootdir
        if mwkname is None:
            a0 = glob.glob(os.path.join(rootdir, "*.mwk"))
            assert len(a0) == 1, "Not just one .mwk in rootdir %s, found: %s" % (rootdir, a0)
            mwkname = a0[0]
        self.mwkname = mwkname
        self.basename = os.path.basename(mwkname.replace(".mwk", ""))
        if not os.path.exists(
            self.mwkname
        ):  # note mwk can be changed to a directory (caching) by MWorks reader code
            raise RuntimeError("mwk not found: %s" % self.mwkname)

        self.outdir_stacks = os.path.join(self.rootdir, "AnalysisStacks")
        os.makedirs(self.outdir_stacks, exist_ok=True)
        self.outdir_figs = os.path.join(self.rootdir, "figures")
        os.makedirs(self.outdir_figs, exist_ok=True)

        self.mwkh5name = self.mwkname.replace(".mwk", ".h5")
        self.stimh5name = self.mwkname.replace(".mwk", "-stims.h5")

        # generate the h5 file from mwk
        print("Converting mwk to h5 if missing or if redoCached...", end="")
        if redoCached:
            exist_delete = True
        else:
            exist_delete = "skip_create"
        mworksbehavior.mwk_io.mwk_to_h5(mwkname, keep_system_vars=False, exist_delete=exist_delete)
        print("done.")

    def parse_the_mwk(self, **kwargs):
        """Extra args are passed to mwkfiles.RetinotopyMap1MWKFile init
        Notes
            this is a separate call to improve debugging and also to allow params to be passed here"""
        self.mwf = mwkfiles.RetinotopyMap1MWKFile(self.mwkh5name, **kwargs)
        self.counterPreVar = "counterNPre"  # may change if we change RM1MwkFile

    def write_stack_for_bootstrap(
        self, im, base_ns_fromstim=None, stim_ns_fromstim=None, map_smooth_sigma=None, print_status=True
    ):
        """Compute/write image stack (nStims,nReps,nRows,nCols)
        Averages across timepoints.
        Uses nonrandom stacks

        Notes:
            df is stim-base on each frame.  dfof is df/f0.  f0 is the baseline average over _all_ frames
        Returns:
            (allrep_df,allrep_dfof) And writes some stacks to root_dir/Analysis
        """
        mwf = self.mwf
        if print_status:
            print("Computing stacks for bootstrap... ", end="")

        # reshape by stim block
        nframes, nrows, ncols = im.shape
        nreps = int(im.shape[0] / (mwf.nframes_stim * mwf.nstim))
        mat = im.reshape([nreps * mwf.nstim, mwf.nframes_stim, nrows, ncols])

        # take means over timepts
        base_ns = base_ns_fromstim + mwf.constS.counterNPre
        stim_ns = stim_ns_fromstim + mwf.constS.counterNPre
        baseMat = np.mean(mat[:, base_ns, :, :], axis=1)
        stimMat = np.mean(mat[:, stim_ns, :, :], axis=1)

        # now pull out all the frames for each level
        levels = np.unique(mwf.stimDf[mwf.levelVar])
        allrepmat = np.zeros((mwf.nstim, nreps, 2, nrows, ncols), dtype="f4")
        for (iL, tL) in enumerate(levels):
            desIx = (
                mwf.stimDf[mwf.levelVar].values == tL
            )  # values turns into np.array - boolean comparison with Series gives silent bugs pd 0.18 171222 MH
            allrepmat[iL, :, 0, :, :] = baseMat[desIx, :, :]
            allrepmat[iL, :, 1, :, :] = stimMat[desIx, :, :]

        # find df, dfof.  See notes above
        f0 = np.mean(allrepmat[:, :, 0, :, :], axis=1)
        if map_smooth_sigma is not None:
            f0 = ndimage.gaussian_filter(f0, sigma=map_smooth_sigma, order=0)
        df = allrepmat[:, :, 0, :, :] - allrepmat[:, :, 1, :, :]
        if map_smooth_sigma is not None:
            print("smoothing (sigma=%.3g)..." % map_smooth_sigma)
            df = ndimage.gaussian_filter(df, sigma=[0, 0, map_smooth_sigma, map_smooth_sigma], order=0)
        dfof = df / f0[:, nax, :, :] * 100.0

        # write outputs
        os.makedirs(self.outdir_stacks, exist_ok=True)
        if print_status:
            print("Writing to %s... " % self.outdir_stacks, end="")
        io.imsave(os.path.join(self.outdir_stacks, "allrep_df.tif"), df.astype("float32"))
        io.imsave(os.path.join(self.outdir_stacks, "allrep_dfof.tif"), dfof.astype("float32"))
        if print_status:
            print("done.")
        return (df, dfof)


def process_filenames(rootdir, mwkname=None):
    """Deprecated -- Create filenames, check all paths exist, create if necessary.

    Args:
        rootdir: directory where imageFrames.tif can be found.
        mwkname: string or None (default): path to mwk file.  If None, use the single .mwk file in rootdir.

    Returns:
        (outname, stimoutname, figoutname, mwkname)

    """
    warnings.warn("Deprecated function, use RetinoMapProcessor() instead")
    if not os.path.isdir(rootdir):
        raise RuntimeError("rootdir not found: %s" % rootdir)

    if mwkname is None:
        a0 = glob.glob(os.path.join(rootdir, "*.mwk"))
        assert len(a0) == 1, "Not just one .mwk in rootdir %s, found: %s" % (rootdir, a0)
        mwkname = a0[0]

    if not os.path.exists(mwkname):  # note mwk can be changed to a directory (caching) by MWorks reader code
        raise RuntimeError("mwk not found: %s" % mwkname)

    outname = mwkname.replace(".mwk", ".h5")
    stimoutname = mwkname.replace(".mwk", "-stims.h5")
    figoutname = mwkname.replace(".mwk", "-map.pdf")

    ld = locals()

    return Namespace(**{k: ld[k] for k in ["outname", "stimoutname", "figoutname", "mwkname"]})


def read_imageframes(root_dir, filename="imageFrames.tif", downscaleFactor=None, fov_frames=None):
    """Read widefield raw data from disk, with optional downscaling

    Args:
        downscaleFactor: factor to divide each of the row and col dimensions, or None 
            to skip downscaling.  Must evenly divide both the row and col dims.
            Note we force same factor for both row, col dims to preserve aspect ratio.
        fov_frames: {*None*, 'all', sequence of frame numbers}: if None, don't compute fov
            if 'all', use all frames for FOV.  If a sequence average over the specified frames.
            Note FOV is computed on full-res images
    Returns:
         im
         (im, fov)   if fov_frames is not None
         im: the full stack
         fov: the fov image, full resolution even if downscale is set

    Notes:
        180610 MH: this requires reading the whole stack into memory.  Looks like there
        are routines in Caiman that can avoid this if it becomes an issue

    """
    fname = os.path.join(root_dir, filename)
    if os.stat(fname).st_size / 1e9 > 5:
        warnings.warn("Reading tiff stack greater than 5GB: be careful with swapping")
    im = io.imread(fname, plugin="tifffile")

    if downscaleFactor is None:
        downscaleFactor = 1
    if im.nbytes / 1e9 / downscaleFactor ** 2 > 1e9:
        warnings.warn("stack after downscaling is > 1GB.  Downscale more?")

    if fov_frames is not None:
        if fov_frames == "all":
            fov_frames = np.arange(im.shape[0])
        fov = im[fov_frames, :, :].mean(axis=0)

    if downscaleFactor > 1:
        im = ptMH.image.downscale_in_chunks(im, downscale_tuple=(1, downscaleFactor, downscaleFactor))

    if fov_frames is None:
        return im
    else:
        return (im, fov)


def write_average_stacks_nonrandom(root_dir, im, nstim, nframes_stim, print_status=True):
    """Compute and write averages across repetitions (for sequential presentations)

    Args:
        nstim: number of stimuli
        nframes_stim: number of frames per stim

    Returns:
        (allstimavg,df,dfof) And writes some stacks to root_dir/Analysis
    """

    if print_status:
        print("Computing average stacks... ", end="")
    (nframes, nrows, ncols) = im.shape
    nframes_per_rep = nstim * nframes_stim
    nreps = int(nframes / nframes_per_rep)
    assert nreps * nframes_per_rep == nframes, "not an even number of frames?  may need truncating?"
    stNL = [x * nframes_stim for x in r_[:nstim]]

    allstimavg = im.reshape((nreps, nframes_per_rep, nrows, ncols)).mean(axis=0)

    # use last 10 frames of each stim as f0
    baseNs = np.concatenate([r_[x + nframes_stim - 10 : x + nframes_stim] for x in stNL])
    f0 = allstimavg[baseNs, :, :].mean(axis=0)
    # find df, dfof
    df = allstimavg - f0
    dfof = df / f0 * 100

    # write stacks
    outdir = os.path.join(root_dir, "analysisStacks")
    if print_status:
        print("Writing to %s... " % outdir, end="")

    io.imsave(os.path.join(outdir, "allstimavg.tif"), allstimavg.astype("float32"))
    io.imsave(os.path.join(outdir, "df.tif"), df.astype("float32"))
    io.imsave(os.path.join(outdir, "dfof.tif"), dfof.astype("float32"))
    if print_status:
        print("done.")

    return (allstimavg, df, dfof)


def plot_stim_maps(
    allstimavg,
    mwf,
    dfofRange=[-1, 2.0],
    base_range=None,
    stim_range=None,
    title_field=None,
    dfof_cmap="PuOr_r",
    smoothing=0,
    plot_grid=True,
):
    """Plot dfof average images.

        Notes:
            - works for random presentations.  Requires the mwk file.

        Args:
            allstimavg: stack, shape (nstims,nframes_per_stim, nrows, ncols).  Average stack, after repetitions of same stim
                have been averaged.  write_average_stacks_random() reshapes the images into this array.
            mwf: Retinotopy MWFile object
            base_range: seq or None (default): the frames within each repetition to use as baseline (non-stim).  None: autocalc, use all
            stim_range: seq or None (default): the frames within each repetition to use as stimulus.  None: autocalc, use all
            dfof_cmap: string, 'PuOr_r' (default): matplotlib colormap to use.  Other good candidates: RdBu_r, Spectral_r
            smoothing: Apply a gaussian blur to stim figure panels. 0 = No smoothing.

        Returns:
            fig: fig handle
    """

    assert len(allstimavg.shape) == 4, "allstimavg should be shape (nstims,nfr_per_stim, nrows, ncols)"
    cS = mwf.constS  # shortcut
    # levelDf: sorted by stimPySelLevel, index is reset to match
    levelDf = mwf.stimDf.drop_duplicates().sort_values(mwf.levelVar).reset_index(drop=True)

    nsp = int(np.ceil(np.sqrt(mwf.nstim + 2)))  # +2: fov and stims
    sp = (nsp, nsp)
    fig = plt.figure(figsize=r_[nsp, nsp] * r_[1, 0.75] * 4)

    if base_range is None:
        base_range = r_[0 : cS.counterNPre]
    if stim_range is None:
        stim_range = r_[cS.counterNStim : mwf.nframes_stim]

    # Apply smoothing to stim panels
    smoothing = smoothing
    s1 = allstimavg
    s1 = ndimage.gaussian_filter(s1, sigma=smoothing, order=0)

    def _fixup_imgax(cbticks=None):
        plt.grid(which="major", axis="both", linewidth=0.25)
        if not plot_grid:
            plt.grid(False)
        plt.tick_params(axis="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        cbar = plt.colorbar(ticks=cbticks)
        plt.gca().set_aspect("equal")
        return cbar

    with sns.axes_style("whitegrid"):
        for iS in range(mwf.nstim):
            plt.subplot(*sp, iS + 1)
            b0 = s1[iS, base_range, :, :].mean(axis=0)
            s0 = s1[iS, stim_range, :, :].mean(axis=0)
            dfof0 = np.squeeze((s0 - b0) / b0 * 100)
            plt.imshow(dfof0, clim=dfofRange, cmap=dfof_cmap)
            cbar = _fixup_imgax(cbticks=np.fix(np.sort(np.hstack((dfofRange, 0)))))
            cbar.ax.set_ylabel("$df/F_0$")
            tStr = "stim %d" % iS
            if title_field is not None:
                tStr = tStr + ": " + "%.3g" % levelDf.loc[iS, title_field]
                if title_field[-2:].lower() == "mw":
                    tStr = tStr + "mW"
            plt.title(tStr)

    # fov image
    ax = plt.subplot(*sp, mwf.nstim + 1)
    fov = allstimavg.mean(axis=1).mean(axis=0)
    plt.imshow(fov, cmap="gray")
    _fixup_imgax()

    ax = plt.subplot(*sp, mwf.nstim + 2)

    def plot_stim_azel(azName, elName, textcolor="k"):
        for (tIndex, tR) in levelDf.iterrows():
            az = getattr(tR, azName)
            el = getattr(tR, elName)
            plt.plot(az, el, marker="o", markersize=20, mfc="0.9", mec="w", color=None)
            plt.text(
                az,
                el,
                "%d" % tR[mwf.levelVar],  # index can change/be resorted
                va="center",
                ha="center",
                color=textcolor,
            )
        ax.set_xlim([-50, 50])
        plt.xlabel("az (deg)")
        ax.set_ylim([-30, 30])

    def write_text_level_changes(levelDf):
        s0 = levelDf.std()
        changing_vars = s0.index[s0 > 0]
        #changing_vars = changing_vars[changing_vars != mwf.levelVar]  # no need to remove levelvar
        ax.text(0.1, 0.95, str(changing_vars.to_numpy()), transform=ax.transAxes, va="top")
        for (iR, tR) in enumerate(levelDf.iterrows()):
            ax.text(
                0.1,
                0.95 - (iR + 1) * 0.08,
                levelDf.loc[iR, changing_vars].to_numpy(),
                transform=ax.transAxes,
                va="top",
            )

    # plot stim info, specific to type of file, ChRMap or Retinotopy
    if type(mwf) == mwkfiles.ChRMap1MWKFile:
        pass  # text level changes only
    elif type(mwf) == mwkfiles.RetinotopyMap1MWKFile:
        plot_stim_azel("tStimAzimuthDeg", "tStimElevationDeg")
    elif type(mwf) == mwkfiles.RetinotopyMap2StimMWKFile:
        if np.any(levelDf.tStim1Contrast > 0):
            plot_stim_azel("tStim1AzimuthDeg", "tStim1ElevationDeg", "k")
        if np.any(levelDf.tStim2Contrast > 0):
            plot_stim_azel("tStim2AzimuthDeg", "tStim2ElevationDeg", "r")
    write_text_level_changes(levelDf)

    # suptitle same for both
    outR = [None] * 2
    for (iR, tR) in enumerate((base_range, stim_range)):
        if np.all(np.diff(tR) == 1):
            outR[iR] = "r_[%d:%d]" % (tR[0], tR[-1] + 1)
    fig.suptitle(r"$\bf{%s}$: base %s, stim %s" % (mwf.basename.replace("_", "\_"), outR[0], outR[1]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_indv_stim_maps(
    allstimavg,
    mwf,
    dfofRange=[-1, 2.0],
    base_range=None,
    stim_range=None,
    stim_id=0,
    dfof_cmap="PuOr_r",
    smoothing=0,
):
    """Plot dfof average images of individual stims.

        Notes:
            - works for random presentations.  Requires the mwk file.

        Args:
            allstimavg: stack, shape (nstims,nframes_per_stim, nrows, ncols).  Average stack, after repetitions of same stim
                have been averaged.  write_average_stacks_random() reshapes the images into this array.
            mwf: Retinotopy MWFile object
            base_range: seq or None (default): the frames within each repetition to use as baseline (non-stim).  None: autocalc, use all
            stim_range: seq or None (default): the frames within each repetition to use as stimulus.  None: autocalc, use all
            stim_id: specific stim location to focus analysis
            dfof_cmap: string, 'PuOr_r' (default): matplotlib colormap to use.  Other good candidates: RdBu_r, Spectral_r
            smoothing: Apply a gaussian blur to stim figure panels. 0 = No smoothing.

        Returns:
            fig: fig handle
    """

    assert len(allstimavg.shape) == 4, "allstimavg should be shape (nstims,nfr_per_stim, nrows, ncols)"
    cS = mwf.constS  # shortcut

    nsp = int(np.ceil(np.sqrt(mwf.nstim + 2)))  # +2: fov and stims
    sp = (nsp, nsp)
    fig = plt.figure()

    if base_range is None:
        base_range = r_[0 : cS.counterNPre]
    if stim_range is None:
        stim_range = r_[cS.counterNStim : mwf.nframes_stim]

    # Apply smoothing to stim panels
    smoothing = smoothing
    s1 = allstimavg
    s1 = ndimage.gaussian_filter(s1, sigma=smoothing, order=0)

    def _fixup_imgax(cbticks=None):
        plt.grid(False)
        plt.tick_params(axis="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        cbar = plt.colorbar(ticks=cbticks)
        plt.gca().set_aspect("equal")
        return cbar

    iS = stim_id
    b0 = s1[iS, base_range, :, :].mean(axis=0)
    s0 = s1[iS, stim_range, :, :].mean(axis=0)
    dfof0 = np.squeeze((s0 - b0) / b0 * 100)
    plt.imshow(dfof0, clim=dfofRange, cmap=dfof_cmap)
    cbar = _fixup_imgax(cbticks=np.fix(np.sort(np.hstack((dfofRange, 0)))))
    cbar.ax.set_ylabel("$df/F_0$")
    plt.title("stim %d" % iS)

    return fig


def write_average_stacks_random(
    outdir, im, mwf, preFrameStartFrac, preFrameEndFrac, counterPreVar="counterNPre", print_status=True
):
    """Compute and write averages across repetitions (for random presentations)
    Averages across stimPySelLevel levels.

    Args:
        root_dir:
        im: image stack
        mwk: mwk file, instance of RetinotopyMap1MWKFile
        preFrameStartFrac: fraction of preframes that are start of win for f0
        preFrameEndFrac: fraction of preframes that are end of win for f0
        counterPreVar: 'counterNPre' (retino default), variable name that lists number of frames before stim (for calc baseline)

    Returns:
        (allstimavg,df,dfof) And writes some stacks to root_dir/Analysis
        first axis of each is ordered by sorted stimPySelLevel levels (i.e. 0,1,2,3 etc)

    Todo:
        This prestim frame / frac calculation is a mess and should be rethought.  Probably need a range and some other way to do 
        fractions, maybe save prevar in the counter mwkfile mixin
    """

    if print_status:
        print("Computing average stacks... ", end="")
    nreps = int(im.shape[0] / (mwf.nframes_stim * mwf.nstim))

    # reshape by stim block
    nframes, nrows, ncols = im.shape
    mat = im.reshape([nreps * mwf.nstim, mwf.nframes_stim, nrows, ncols])

    # take means based on stim number
    levels = np.unique(mwf.stimDf[mwf.levelVar])
    allstimavg = np.zeros((mwf.nstim, mwf.nframes_stim, nrows, ncols), dtype="f4")
    for (iL, tL) in enumerate(levels):
        desIx = (
            mwf.stimDf[mwf.levelVar].values == tL
        )  # values turns into np.array - boolean comparison with Series gives silent bugs pd 0.18 171222 MH
        tM = np.mean(mat[desIx, :, :, :], axis=0)
        allstimavg[iL, :, :, :] = tM

    # f0
    stN = int(np.floor(mwf.constS[counterPreVar] * preFrameStartFrac))
    endN = int(np.ceil(mwf.constS[counterPreVar] * preFrameEndFrac))
    f0 = allstimavg[:, stN:endN, :, :].mean(axis=1)
    f0 = f0[:, np.newaxis, :, :]

    # find df, dfof
    df = allstimavg - f0
    dfof = df / f0 * 100.0
    dfof = dfof.reshape(mwf.nstim * mwf.nframes_stim, nrows, ncols)

    # write stacks
    os.makedirs(outdir, exist_ok=True)
    if print_status:
        print("Writing to %s... " % outdir, end="")

    io.imsave(os.path.join(outdir, "allstimavg.tif"), allstimavg.astype("float32"))
    io.imsave(os.path.join(outdir, "df.tif"), df.astype("float32"))
    io.imsave(os.path.join(outdir, "f0.tif"), f0.astype("float32"))
    io.imsave(os.path.join(outdir, "dfof.tif"), dfof.astype("float32"))
    if print_status:
        print("done.")

    locs = locals()
    return Namespace(**{k: locs[k] for k in ["allstimavg", "df", "dfof", "stN", "endN", "levels"]})


class RetinoMapPlotter:
    def __init__(self, allstimavg, mwf, base_ns=None, stim_ns=None, smooth_sigma=4, signalIsNeg=True):
        """Setup.
        Args:
            sigIsNeg: True for hemodynamic imaging, False for fluo/flavo
            base_ns, stim_ns: if you don't know the best ones yet, leave this as none.
                Typical sequence: __init__(), fig_timecourse_response(), choose best base/stim range,
                then call update_base_stim_range(), then proceed with other maps
                Note: these are relative to the start of each stim block (i.e. relative to the start of
                counterNPre) not relative to stim onset.

        Notes:
            Can also be used for ChrMap mwk files
            leave dfof sign as on input.  If signalIsNeg, handle in plotting functions
            Do smoothing and dfof here.  If we need to resmooth, perhaps add another fn? or just reinstantiate

        Todo: should add a way to get the high-res FOV

        """
        self.base_ns = base_ns
        self.stim_ns = stim_ns
        self.allstimavg = allstimavg
        self.mwf = mwf
        self.signalIsNeg = signalIsNeg
        if not self.signalIsNeg:
            warnings.warn("This code is not debugged well for positive signals!! check below")
        (self.nStim, self.nFr, self.nY, self.nX) = np.shape(self.allstimavg)

        self.allDfof = None  # will be updated
        # levelDf: sorted by stimPySelLevel, index is reset to match
        self.levelDf = mwf.stimDf.drop_duplicates().sort_values(mwf.levelVar).reset_index(drop=True)

        self.smooth_sigma = smooth_sigma

    def _shade_vert_range(self, trange, color=None):
        ax = plt.gca()
        ylim = ax.get_ylim()

        asH = plt.axvspan(trange.min(), trange.max(), alpha=0.3)
        if color is not None:
            asH.set_color(color)
        plt.plot(trange, [ylim[0] + 0.1 * np.ptp(ylim)] * len(trange), ".", color=asH.get_facecolor())

    def fig_timecourse_response(self, tcXRange=None, tcYRange=None):
        """
        Args:
           tcXRange,YRange, sequences, length 2: range of X (Y) pixels to use.  None means use all
        """
        gs = mpl.gridspec.GridSpec(2, 2)
        fovSm = self.allstimavg.mean(axis=(0, 1))
        preNs = getattr(self.mwf.constS, self.mwf.preVar)
        def _do_shade():
            if self.base_ns is not None:
                self._shade_vert_range(self.base_ns - preNs)
            if self.stim_ns is not None:
                self._shade_vert_range(
                    self.stim_ns - preNs, color=sns.color_palette()[1]
                )

        fig = plt.figure(figsize=10 * r_[1, 0.75])
        (nStim, nTp, nY, nX) = self.allstimavg.shape
        xs = r_[0:nTp] - preNs
        if tcXRange is None:
            tcXRange = r_[0, nX]
        if tcYRange is None:
            tcYRange = r_[0, nY]
        tcXRange = a_(tcXRange)
        tcYRange = a_(tcYRange)
        tcXNs = r_[tcXRange[0] : tcXRange[1]]
        tcYNs = r_[tcYRange[0] : tcYRange[1]]
        tc = self.allstimavg[:, :, tcYNs, :][:, :, :, tcXNs].mean(axis=(2, 3))

        ax = plt.subplot(gs[0, 0])
        plt.plot(xs, tc.T)
        plt.legend(["stim %d" % d for d in range(self.mwf.nstim)])
        plt.title("average timecourse by stim")
        _do_shade()

        ax = plt.subplot(gs[0, 1])  # tc plot, this one uses a fancy title to give ranges
        plt.plot(xs, tc.mean(axis=0))
        plt.plot(xs, ptMH.math.smooth_lowess(tc.mean(axis=0)))

        def _rtomm(_range):
            # simplify lists of numbers into a min:max format
            if _range is None:
                return "None"
            if np.all(np.diff(_range) == 1):
                return "[%d:%d]" % (_range.min(), _range.max() + 1)  # +1: use the Python range/slice format
            else:
                return repr(_range)

        plt.title(
            "average tc over all stim, base,stim_ns: %s,%s (- pre fr %d)"
            % (_rtomm(self.base_ns), _rtomm(self.stim_ns), preNs)
        )
        _do_shade()

        ax = plt.subplot(gs[1, 0])
        plt.imshow(fovSm, cmap="gray")
        plt.plot(
            tcXRange[[0, 1, 1, 0, 0]],
            tcYRange[[0, 0, 1, 1, 0]],
            lw=3,
            ls="--",
            color=sns.color_palette("Paired")[3],
        )
        plt.title("FOV: green square shows pixels used")
        return fig

    def update_base_stim_range(self, base_ns, stim_ns):
        """Sets attributes: base_ns, stim_ns, allDfof"""
        self.base_ns = base_ns
        self.stim_ns = stim_ns

        # compute these only if base/stim range specified, else leave as None as set in __init__()
        if not (self.base_ns is None or self.stim_ns is None):
            self.allDfof = np.zeros((self.nStim, self.nY, self.nX), "f8")
            for iS in range(self.nStim):
                self.b0 = self.allstimavg[iS, self.base_ns, :, :].mean(axis=0)
                self.s0 = self.allstimavg[iS, self.stim_ns, :, :].mean(axis=0)
                dfof0 = np.squeeze((self.s0 - self.b0) / self.b0 * 100)

                s1 = ndimage.gaussian_filter(dfof0, sigma=self.smooth_sigma, order=0)
                self.allDfof[iS, :, :] = s1

    def fig_activation_contours(self, pctThresh=50, maxPlotPctile=99):
        nP = int(np.ceil(np.sqrt(self.nStim)))
        fig = plt.figure(figsize=12 * r_[1, 0.75])
        gs = mpl.gridspec.GridSpec(nP, nP)

        topPct = np.percentile(np.abs(self.allDfof), maxPlotPctile)  # for clim only
        for iS in range(self.nStim):
            ax = plt.subplot(gs[iS])
            ax.grid(lw=0.5)
            tFr = self.allDfof[iS, :, :]
            plt.imshow(tFr, cmap="RdBu_r", clim=topPct * r_[-1, 1])
            plt.colorbar()

            # find fwhm contours
            if self.signalIsNeg == True:  # hemo
                cVal = tFr.min()
                cStr = "min"
            if self.signalIsNeg == False:  # gcamp
                cVal = tFr.max()
                cStr = "max"

            plt.contour(tFr, [cVal * pctThresh / 100])
            plt.title("stim %d, %s %.3g%%" % (iS, cStr, cVal))

        fig.suptitle("%s - contours: %d%% %s" % (self.mwf.basename, pctThresh, cStr))

    def fig_detailed_contours_with_fov(self, contourPctiles=[50, 80], filter_sigma=None):
        """Plot all contours and centroids on one plot
        Args:
            cmPctileL: list of two percentiles of center mass to plot as lines.
            filter_sigma: None, or the sigma of a gaussian filter.  Have tried sigma 0.7 before. Default None"""

        fig = plt.figure(figsize=15 * r_[1, 0.75])
        plt.imshow(self.b0, cmap="gray")
        ax = plt.gca()
        ax.grid(lw=0.5)
        mHL = []
        assert len(contourPctiles) == 2, "check contour pctile length"
        assert np.diff(contourPctiles) > 0, "contourPctileL must be increasing"
        for iS in range(self.nStim):
            tImg = self.allDfof[iS, :, :].copy()

            if self.signalIsNeg == True:
                tVal = tImg.min()
                tArgN = tImg.argmin()
                tCompF = np.less
            elif self.signalIsNeg == False:
                tVal = tImg.max()
                tArgN = tImg.argmax()
                tCompF = np.greater

            if filter_sigma is not None:
                tImg = ndimage.gaussian_filter(tImg, filter_sigma)

            # +: center mass, 1st percentile thresholded
            tImgMasked = tImg.copy()
            tImgMasked[tImgMasked > tVal * contourPctiles[0] / 100.0] = 0.0
            (y, x) = ndimage.measurements.center_of_mass(tImgMasked)
            mH = plt.plot(x, y, "+", ms=15, mew=4)
            tColor = mH[0].get_color()

            # x: minimum pt
            (y, x) = np.unravel_index(tArgN, tImg.shape)
            mH = plt.plot(x, y, "x", ms=5, mew=1, color=tColor)
            mHL.append(mH)

            # contour lines
            plt.contour(
                tImg, [tVal * contourPctiles[0] / 100], colors=tColor, linewidths=0.5, linestyles="--"
            )
            plt.contour(tImg, [tVal * contourPctiles[1] / 100], colors=tColor, linewidths=1.0, linestyles="-")

            # circle: center of mass, 2nd percentile thresholded *ignoring intensity above thresh*
            tImgBin = tCompF(tImg, tVal) * contourPctiles[1] / 100
            (y, x) = ndimage.measurements.center_of_mass(tImgBin)
            mH2 = plt.plot(x, y, "o", ms=15, mew=1.0, mfc=None, color=mH[0].get_color(), alpha=0.5)

        plt.legend([mH[0] for mH in mHL], ["stim %d" % iS for iS in range(self.nStim)], frameon=True)
        plt.title(
            "smoothed df/f images: +: c.mass > {:g}, x: min, o: c.mass > {:g}".format(
                *a_(contourPctiles, dtype="f8")
            )
        )

    def fig_hsv_colormap(self, saturationExponent=1.5):
        """
        Args:
            saturationExponent:  may wish to set to 2-4 to plt color maps alone (w/out gray fov), to reduce small sat values
                1.5 exponent seems to work well with gray fov background
        Notes:
            normalizes each activation image independently
        """

        fig = plt.figure(figsize=6 * r_[1, 0.75] * r_[2, 2])
        gs = mpl.gridspec.GridSpec(2, 3)
        ax = plt.subplot(gs[0:2, 0:2])
        ax.grid(lw=0.5)
        ax.set_aspect("equal")

        # first construct the color map as in schuett, 2002
        normDfof = _normarr(
            self.allDfof, axis=0
        )  # normalize each image independently, and invert so min is ma
        if self.signalIsNeg:
            normDfof = 1 - normDfof
        maxColorImg = np.argmax(normDfof, axis=0) / self.nStim
        maxIntensImg = np.max(np.abs(normDfof), axis=0)
        sat = maxIntensImg ** saturationExponent
        rgb = color.hsv2rgb(np.stack((maxColorImg, sat, sat), axis=2))

        # fov image, from background
        plt.imshow(self.b0, cmap="gray")
        alpha = maxIntensImg[:, :, n_]
        rgba = np.concatenate((rgb, alpha), axis=2)
        plt.imshow(rgba)

        # now make a legend on a different axis
        cvec = np.arange(self.nStim, dtype="f8") / self.nStim
        colortups = color.hsv2rgb(np.stack((cvec, cvec * 0 + 1, cvec * 0 + 1), axis=-1)[n_, :, :])
        ax = plt.subplot(gs[0, 2])
        ax.set_aspect("equal")
        dfu = self.mwf.stimDf.drop_duplicates()
        for (tIndex, tR) in dfu.iterrows():
            lv = int(tR[self.mwf.levelVar])
            plt.plot(
                tR.tStimAzimuthDeg,
                tR.tStimElevationDeg,
                marker="o",
                markersize=20,
                mec="w",
                mfc=colortups[0, lv, :],
            )
            plt.text(
                tR.tStimAzimuthDeg,
                tR.tStimElevationDeg,
                "%d" % lv,  # index can change/be resorted
                va="center",
                ha="center",
                fontsize=14,
            )
        ax.set_xlim([-50, 50])
        plt.xlabel("az (deg)")
        ax.set_ylim([-30, 30])

        plt.suptitle(self.mwf.basename)
        plt.tight_layout()
