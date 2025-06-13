"""This module runs Caiman to extract cells and timecourses from 2p data

TODOs
- 180508 MH: capture output from run_motion_correct and save to a log file
- fix all the warnings
- 180508 - now using caiman start/stop cluster commands, later may wish to write our own

"""

from builtins import zip
from builtins import str
from builtins import map
from builtins import range

import multiprocessing as mp
import packaging; from packaging.version import Version
import pytoolsMH as ptMH

import numpy as np
import sys, os
r_ = np.r_
a_ = np.asarray

import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

import caiman as cm
import os
import time
import glob
import logging, logging.config
import tempfile
import scipy.signal as ss
from argparse import Namespace
from skimage.external import tifffile
import yaml
import contextlib
import warnings
import joblib

# plotting
import IPython, ipywidgets
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


from caiman.utils.visualization import plot_contours
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.utilities import detrend_df_f

from .consts import DataDir2p

## check versions
assert(Version(ptMH.__version__) >= Version('2018.05')) # for new fns
assert (Version(cm.__version__) >= Version('2018.05')), 'conflict: use histedlab CaImAn, and check version'


# logging
logger = logging.getLogger(__name__)
import asyncio
asyncio.get_event_loop().set_debug(False)  # no need to see asyncio's debugging output


######################################################################




class CaimanAnalyze():
    """Limited class that loads outputs and does extra analysis.
    Each analysis method should be independent.  At some point we may wish to cache stacks/tifs in memory..."""

    def __init__(self, fname):
        copyAttribL = ['origstackname', 'basedir', 'mc_data']
        with CaimanRun(fname, showPlots=False) as cr: # does not do any computation on init
            cr.load_motion_correct_output()
            for tA in copyAttribL:
                setattr(self, tA, getattr(cr, tA))

            # copy and rename a few fields
            self.mc_tif_name = cr._out_mc_tif_name
            self.mc_data_name = cr._out_mc_data_name
            self.cnm_data_name = cr._out_cnm_data_name
            self.run_outdir = cr.outdir

        self.extra_outdir = os.path.join(self.basedir, 'analysisExtra')
        os.makedirs(self.extra_outdir, exist_ok=True)

        # use DataDir2p for loading
        self.dd2p = DataDir2p(self.basedir)
        self.cnmf_data = self.dd2p.get_caiman_results()


    def write_fov(self):
        """Load stack (default mc) and write a FOV"""
        warnings.warn('Loading full motion corrected stack: %s' % self.mc_tif_name)
        mov = cm.load(self.mc_tif_name)
        fov = mov.mean(axis=0)
        tifffile.imsave(os.path.join(self.extra_outdir, 'fov-float.tif'), fov.astype('f4'))
        plt.imsave(os.path.join(self.extra_outdir, 'fov.png'), fov, cmap='gray')


    def write_background_images(self, doWrite=True):
        """Write the background components as images"""
        b = self.cnmf_data.b
        d1 = self.cnmf_data.d1
        d2 = self.cnmf_data.d2
        nBackgComp = b.shape[1]

        nP = int(np.ceil(np.sqrt(nBackgComp+1)))
        gs = mpl.gridspec.GridSpec(nP, nP)
        fig = plt.figure(figsize=r_[1, 0.75] * nP*6)

        for iC in range(nBackgComp):
            ax = plt.subplot(gs[iC])
            im = b[:, iC].reshape((d1, d2))
            plt.imshow(im, cmap='viridis', clim=[0, np.percentile(b[:, iC], 99)])
            plt.grid(lw=0.25)
            #ax.invert_xaxis()
            #ax.invert_yaxis()
            plt.title('backg comp %d' % iC)
            plt.colorbar()

        # draw traces
        ax = plt.subplot(gs[nBackgComp])
        plt.plot(self.cnmf_data.f.T)

        plt.tight_layout()
        if doWrite:
            fig.savefig(os.path.join(self.extra_outdir, 'fig-backg-components.pdf'))


class CaimanRun():
    """Does a full caiman run, saving intermediate images and all outputs to our set locations.

    Inputs: The filename of imageFrames.tif, or its root directory.
    Outputs: analysisCaiman directory

    Call as a context manager/with statement:

    with CaimanRun(fname) as cr:
        cr.dostep1
        cr.dostep2

    This ensures cleanup happens properly.

    Args (to __init__()):
        debugKeepOnlyNFr:

    Attribs:
        workstackfullname: current temp tif, output of each manipulation step, deleted at end; full name with path
        origstackname: source file, imageFrames.tif, no path
        workminval: working movie minimum, used by motion correction to prevent zero values.  Set lazily on first load

    Notes:
        - We don't keep the movie tiff in memory, we read it as needed, since that's what Caiman works with: on disk files
        - Because we're using local computation for now, we start/stop clusters inside each method step.  If we later
          wish to run against a bigger longer running cluster, add separate methods to start/stop and test for running
          inside the methods
        - We define here as many output filenames and other constants as we can, so they are in one place

    """

    def __init__(self, fname, showPlots=False, debugWorkStack=None, debugKeepOnlyNFr=None):
        """
        Args:
            debugKeepOnlyNFr: None or int, if int, chop stack to only this number of frames to speed debugging
            showPlots: boolean; display the plots.  We always write them to disk regardless
        """
        td, tn = os.path.split(fname)
        if tn == '':
            tn = 'imageFrames.tif'
        self.workstackfullname = os.path.join(td, tn)
        self.origstackname = tn
        self.basedir = td
        self._enter_called = False
        self.mc_data = None
        self.debug = False
        if not os.path.exists(self.workstackfullname):
            raise RuntimeError('Cannot find stack: %s' % self.workstackfullname)

        self.tmpdir = tempfile.gettempdir()
        ptMH.paths.check_min_gb(self.tmpdir, mingb=0.5)

        # outputs, external or desired outputs
        self.outdir = os.path.join(self.basedir, 'analysisCaiman')
        os.makedirs(self.outdir, exist_ok=True)
        self._outlogdir = os.path.join(self.outdir, 'logs')
        os.makedirs(self._outlogdir, exist_ok=True)
        self._out_mc_tif_name = os.path.join(self.outdir, 'imageFrames-mc.tif')  # not always used, see run_motion_correction()
        self._out_mc_data_name = os.path.join(self.outdir, 'results-mc.npz')
        # figs
        self._out_blank_fig_name = os.path.join(self.outdir, 'fig-blank-stats.png') # too many objs for pdf
        self._out_hpxartifact_fig_name = os.path.join(self.outdir, 'fig-hpx-artifact-removal-stats.pdf')
        self._out_cnm1_fig_name = os.path.join(self.outdir, 'cnmf-run1-contours.pdf')
        self._out_cnm_rej_fig_name = os.path.join(self.outdir, 'cnmf-run1-accrej.pdf')
        self._out_cnm2_fig_name = os.path.join(self.outdir, 'cnmf-run2-acccontours.pdf')
        self._out_cnm_data_name_old1 = os.path.join(self.outdir, 'results-analysis.npz')
        self._out_cnm_data_name = os.path.join(self.outdir, 'results-analysis-v2.gz')
        self._out_cnm_trace_fig_name = os.path.join(self.outdir, 'trace_examples.pdf')
        self._log_stdout_name = os.path.join(self._outlogdir, 'stdout.log')
        self._log_logger_name = os.path.join(self._outlogdir, 'pythonlog.log')

        # outputs, internal or logging outputs
        self._mc_mmap_fnameL = None  # set by motion correct, used by CNMF
        self._cnmf_mmap_fname_base = os.path.join(self.tmpdir, 'cnmf_mmap_')  # cnmf completes this and sets _cnmf_mmap_fname
        self._cnmf_mmap_fname = None  #
        self._cnmf_logfile = os.path.join(self.tmpdir, 'cnmf_log.txt')

        self.workminval = None   # will be set below if movie is ever loaded
        self.showPlots = showPlots

        if debugKeepOnlyNFr is not None:
            self.debug = True
            self._movie_drop_and_write_new(keepFrs=r_[:debugKeepOnlyNFr])
            self._update_progressbar('blank')

        self._config_logging()   # setup logging on entry


    def _config_logging(self):
        """Configure logging to a single file"""

        # use a yaml string here to keep all in one file; yaml configuration is far easier than
        # python code.  E.g. backupCount is an arg to RotatingFileHandler, but
        # setLevel and setFormatter must be called as methods.
        configStr = """
        version: 1
        disable_existing_loggers: False

        formatters:
            simple:
                format: "%(asctime)s: %(name)s: %(levelname)s: %(message)s"

        handlers:
            console:
                class: logging.StreamHandler
                level: WARNING
                formatter: simple
                stream: ext://sys.stdout

            debug_file_handler:
                class: logging.FileHandler
                level: DEBUG
                formatter: simple
                filename: {logfile}
                encoding: utf8

        root:
            level: DEBUG
            handlers: [console, debug_file_handler]
        """.format(logfile=self._log_logger_name)
        config = yaml.safe_load(configStr)
        logging.config.dictConfig(config)

        # use sys excepthook to log all exceptions
        #log = logging.getLogger('')  # log exceptions to root logger
        #def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        #    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        #sys.excepthook = log_uncaught_exceptions


    def _update_progressbar(self, curr_step):
        # these are in order
        self._steps = a_(['blank','drop','cluster_setup','mc','mmap_for_cnmf','cnmf1','cnmf2','df/f','cnmf_save'], dtype='O')
        nS = len(self._steps)
        dL = ['CaimanRun: finished ','start',', ','stage ','0','/',str(nS)]
        if not hasattr(self, '_progbar'):
            self._progbar = ipywidgets.FloatProgress(min=0, max=1)
            self._proglabel = ipywidgets.Label(value=''.join(dL))
            IPython.display.display(ipywidgets.HBox([self._proglabel,self._progbar]))

        tSN = np.nonzero(self._steps == curr_step)[0]
        assert (len(tSN)>0), 'bug: curr_step %s not found'%curr_step
        dL[1] = str(curr_step)
        dL[4] = str(int(tSN+1))
        self._proglabel.value = ''.join(dL)
        self._progbar.value = (tSN[0]+1)/nS



    def __enter__(self):
        self._enter_called = True
        self._stdoutFh = open(self._log_stdout_name, 'a')
        self._stdoutFh.writelines(['** Starting **'])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdoutFh.close()
        self._cleanup()


    def _cleanup(self):
        """
        Notes:
            in debug mode, we don't delete temporary stacks
        """
        logger.info('Closing CaimanRun object.  ')

        tmpstacks = glob.glob(os.path.join(self.tmpdir, 'CaimanRun-*'))
        tmpmmap = glob.glob(os.path.join(self.tmpdir, 'cnmf*.mmap'))
        tmpstacks = tmpstacks+tmpmmap
        if self.debug:
            logger.info('Debug mode: not removing %d temporary files' % len(tmpstacks))
        else:
            logger.debug('Removing %d temporary tiff files: %s' % (len(tmpstacks), ', '.join(tmpstacks)))
            for tF in tmpstacks:
                os.remove(tF)



    def drop_artifact_frs(self, nPreStimFr, nPostStimFr, nArtifactFr=1, doPlots=True):
        """For 1p stim imaging: drop frames that contain stim artifacts"""


        (nFr, nX, nY) = ptMH.image.tif_file_get_dims(self.workstackfullname)

        if nArtifactFr > 1:
            raise RuntimeError('Need to add to this code if more than one artifact frame')
        dropFrs = r_[nPreStimFr:nFr:nPreStimFr + nPostStimFr]

        logger.debug('Dropping artifact frames: %s' % dropFrs)

        (origstats,newstats) = self._movie_drop_and_write_new(dropFrs, doStats=True)

        if doPlots:
            fig = plt.figure(figsize=r_[1,0.75]*12)
            gs = mpl.gridspec.GridSpec(2,2)

            # first row, all data
            ax = plt.subplot(gs[0,0])
            plt.plot(origstats.framemean)
            plt.plot(newstats.framemean)
            ax = plt.subplot(gs[0,1])
            plt.plot(origstats.linemean)
            plt.plot(newstats.linemean)

            # second row, zoomed data
            frR = [0,200]
            lineR = [nY*(nPreStimFr-1),nY*(nPreStimFr+2)]
            ax = plt.subplot(gs[1,0])
            plt.plot(origstats.framemean[frR[0]:frR[1]])
            plt.plot(newstats.framemean[frR[0]:frR[1]])
            plt.xlabel('frames')
            ax = plt.subplot(gs[1,1])
            xs = origstats.linexs
            desNs = r_[lineR[0]:lineR[1]]
            plt.plot(xs[desNs], origstats.linemean[desNs])
            plt.plot(xs[desNs], newstats.linemean[desNs])
            linex = r_[lineR[0]:lineR[1]:nY]
            for x in linex:
                ax.axvline(x, lw=0.25)
            plt.xlabel('lines')

            # some adjs
            for ax in fig.get_axes():
                if ax.is_first_col():
                    ax.set_ylabel('pix val')

            fig.savefig(self._out_blank_fig_name)  # lots of points, so save as png
            if not self.showPlots:
                fig.set_visible(False)
                plt.close()

        self._update_progressbar('drop')


    def remove_hpx_artifact_by_imputation(self, skipChecks=False):
        """odd artifact removal for lambda hpx, won't be used that often.

        Notes:
             don't update the progress bar for this one, too infrequent.
        """

        m = cm.load(self.workstackfullname)

        fig,(ax1,ax2) = plt.subplots(1,2)

        plt.sca(ax1)
        (pctLines, lineThresh) = pctile_hist(m,artifactPct=2)

        # now iterate over lines and replace with pix above and below
        [nFr, nY, nX] = m.shape
        nLines = nY * nFr
        mL = m.reshape((nLines, nX))
        nonArtifactPctile = np.percentile(mL[pctLines < lineThresh, :].ravel(), 99)
        pixLImputed = []
        for iL in range(nLines):
            if pctLines[iL] > lineThresh:
                prevLine = mL[iL - 1]
                tLine = mL[iL]
                next2Line = mL[iL + 2]  # use two ahead in case near turnaround
                pixIx = tLine > nonArtifactPctile * 2
                tLine[pixIx] = np.vstack((prevLine, next2Line))[:, pixIx].mean(axis=0)
                pixLImputed.append(np.sum(pixIx))
        plt.title(nonArtifactPctile)
        m2 = mL.reshape((nFr, nY, nX))

        # check if we're running this on a file with no artifacts
        if not skipChecks:
            assert (nonArtifactPctile*2 < lineThresh), 'bug: "real" pix too near threshold: does this file have hpx artifacts?'
            assert (len(pixLImputed) < nLines*0.01), 'bug: more than 1% of lines found; does file have hpx artifact?'
            nearThreshIx = (np.abs(pctLines - lineThresh) < lineThresh*0.08)
            assert (np.sum(nearThreshIx) < 40), 'bug: too many lines right near the threshold: does file have hpx artifact?'

        # replot
        plt.sca(ax2)
        pctile_hist(m2, 2)
        plt.title('nPix imputed: %d, in %d lines' % (a_(pixLImputed).sum(), len(pixLImputed)))

        # now save
        tifffile.imsave(self.workstackfullname, m2)

        fig.savefig(self._out_hpxartifact_fig_name)
        if not self.showPlots:
            fig.set_visible(False)
            plt.close()




    def _movie_compute_pixstats(self, arr):
        """Helper function to compute some pixel statistics: frame mean, line mean, etc
        Notes:
            Can add more stats to output namespace as desired
        """
        m = arr
        (nFr,nY,nX) = m.shape
        return Namespace(framemean=m.mean(axis=2).mean(axis=1),
                         linemean=m.mean(axis=2).reshape((nFr*nY)),
                         framexs=r_[:nFr],
                         linexs=r_[:nFr*nY])



    def _movie_drop_and_write_new(self, dropFrs=None, keepFrs=None, checkMinGb=0.5, doStats=False):
        """Write a new movie after dropping (artifact?) frames.  Always uint16.
        Returns:
            None
        Sets:
            self.workstackfullname
            self._workframemean
            self._worklinemean
        """

        blname = os.path.join(self.tmpdir, 'CaimanRun-pre-motion-correct.tif')  # temp file to pass to motion correction

        # load file.  copy frame after artifact frames into artifact frame
        m = cm.load(self.workstackfullname)
        nFr = m.shape[0]
        if keepFrs is not None:
            assert (dropFrs is None), 'cannot specify both dropFrs and keepFrs'
        else:
            # compute keepFrs from dropFrs
            keepFrs = np.setdiff1d(r_[0:nFr], dropFrs)
        origstats = self._movie_compute_pixstats(m)

        # write back
        logger.info('Writing new temporary stack: %s' % blname)
        m = m[keepFrs,:,:]
        tifffile.imsave(blname, m.astype('uint16'))
        self.workstackfullname = blname

        self.workminval = m.min()

        # compute some means and attach them to this object
        if doStats:
            finalstats = self._movie_compute_pixstats(m)
            return (origstats, finalstats)
        else:
            return None


    def run_motion_correction(self, save_mc_as_tif=False):
        """Run motion correction on a local cluster
        Arguments:
            save_mc_as_tif: boolean, if True, write out the motion-corrected stack to analysisCaiman directory
                else, use a temporary mmap file only
        Notes:
            - We use ipyparallel because multiprocessing sucks for a million small reasons.
            - If ipyparallel becomes difficult, we can add a multiprocessing option here, but we will need to figure
            out how to hold the Pool object so the cluster can be restarted.
            - 180508 - now using caiman start/stop cluster commands, later may wish to write our own
        """

        fname = self.workstackfullname  # tif of stack on disk: blanked if requested

        # set workminval which gets added to stack to remove negatives
        if self.workminval is None:
            self.workminval = -10  # we should never have min values in our movies, so use this

        # start cluster
        #multiprocessing: cannot terminate workers, they will get respawned.  Must terminate with pool object, keeping it around.
        #c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='ipyparallel', n_processes=None, single_thread=False)
        self._update_progressbar('cluster_setup')

        niter_rig = 1  # number of iterations for rigid motion correction  #MH: not even sure _rig params get used
        max_shifts = (6, 6)  # maximum allow rigid shift
        splits_rig = 56  # for parallelization split the movies in  num_splits chuncks across time
        strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
        splits_els = 56  # for parallelization split the movies in  num_splits chuncks across time
        upsample_factor_grid = 4  # upsample factor to avoid smearing when merging patches
        max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
        num_splits_to_process_els = [None]  # MH: not sure why, but default is [7,None] and it looks like the 7 part just gets thrown away.

        self.mc = cm.motion_correction.MotionCorrect(fname, min_mov=self.workminval,
                                                dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                                                splits_rig=splits_rig,
                                                strides=strides, overlaps=overlaps, splits_els=splits_els,
                                                upsample_factor_grid=upsample_factor_grid,
                                                max_deviation_rigid=max_deviation_rigid,
                                                num_splits_to_process_els=num_splits_to_process_els,
                                                shifts_opencv=True, nonneg_movie=True)
        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            self.mc.motion_correct_pwrigid(save_movie=True)  # saves to mmap file

        if save_mc_as_tif:
            # load output mmap and save as tiff
            m_els = cm.load(self.mc.fname_tot_els)
            tifffile.imsave(self._out_mc_tif_name, m_els.astype('uint16'))

        # we need these for cnmf later
        self._bord_px_els = np.ceil(np.maximum(np.max(np.abs(self.mc.x_shifts_els)),
                                         np.max(np.abs(self.mc.y_shifts_els)))).astype(np.int)
        self._mc_mmap_fnameL = self.mc.fname_tot_els

        # save all mc output in case we need it later, or for reloading
        outD = {}
        dropNL = ['dview']  # manually drop these fields
        for tN in dir(self.mc):
            tV = getattr(self.mc, tN)
            if tN[0:2] == '__' or callable(tV) or tN in dropNL:  # drop methods or dunder atttribs
                continue
            else:
                outD[tN] = tV
        outD['bord_px_els'] = self._bord_px_els
        np.savez(self._out_mc_data_name, **outD)
        self.mc_data = Namespace(**outD)

        # kill the cluster we used for motion corr
        cm.stop_server()  # works for ipyparallel.

        self._update_progressbar('mc')



    def load_motion_correct_output(self, debug_file=None):
        """Restores needed parameters for later steps from saved motion correct output.
        Notes:
            This sets up run_cnmf_twostep() to read from a tif, not from an existing mmap file.
        """
        if debug_file is not None:
            readf = debug_file
        else:
            readf = self._out_mc_tif_name
        self.mc_data = Namespace(**dict(np.load(self._out_mc_data_name)))  # load as namespace (dot-access)
        self._mc_mmap_fnameL = [readf]
        self._bord_px_els = self.mc_data.bord_px_els
        self._update_progressbar('mc')


    def run_cnmf_twostep(self, frPerS=None, decayTimeS=None, nBackground=2):
        """Run the CNMF-based cell-extraction and deconvolution, full 2-stage method with automatic bad component rejection

        Notes:
            - if self.mc_data is not None, mc is correctly loaded and this can proceed.
            - assumes cm.save_memmap can work both with mmap files (result of run_mc above) and with tiffs (load_motion_correct_output)
        Sets:
            self.cnm
         """

        if self.mc_data is None:
            raise RuntimeError('No motion correct output found.  Call either run_motion_correction() or load_motion_correct_output() ')

        # first, memory map a new file in order 'C'
        self._cnmf_mmap_fname = cm.save_memmap(self._mc_mmap_fnameL, base_name=self._cnmf_mmap_fname_base, order='C',
                                   border_to_0=self._bord_px_els)  # number of pixels to exclude at border due to motion zeroing/naning them

        # now load the file
        Yr, dims, T = cm.load_memmap(self._cnmf_mmap_fname)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # load frames in python format (T x X x Y)
        logger.info('** Loaded mmap file %s' % self._cnmf_mmap_fname)
        self._update_progressbar('mmap_for_cnmf')

        # cnmf params, for source extraction and deconvolution
        p = 1  # order of the autoregressive system, used only in step 2
        gnb = nBackground  # number of global background components
        merge_thresh = 0.8  # merging threshold, max correlation allowed
        rf = 15  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
        stride_cnmf = 6  # amount of overlap between the patches in pixels
        K = 4  # number of components per patch
        gSig = [4, 4]  # expected half size of neurons
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data, use? 'sparse_nmf')
        #is_dendrites = False  # flag for analyzing dendritic data
        alpha_snmf = None  # sparsity penalty for dendritic data analysis through sparse NMF

        # parameters for component evaluation
        min_SNR = 2.5  # signal to noise ratio for accepting a component
        rval_thr = 0.8  # space correlation threshold for accepting a component
        use_cnn = False
        cnn_thr = 0.8  # threshold for CNN based classifier


        # now run cnmf, on patches
        dview = None  # no cluster, just run single-process
        self.cnm1 = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                        p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf,
                        only_init_patch=False, gnb=gnb, border_pix=self._bord_px_els)

        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            self.cnm1 = self.cnm1.fit(images)

        # now plot contours of these found components, stage 1
        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            Cn = cm.local_correlations(images.transpose(1, 2, 0))
        Cn[np.isnan(Cn)] = 0

        fig = plt.figure()
        crd = plot_contours(self.cnm1.A, Cn, thr_method='nrg', thrnrg=0.9, fontsize=9)
        plt.title('Contour plots of found components')
        fig.set_size_inches(r_[1, 0.75] * 16)
        fig.savefig(self._out_cnm1_fig_name)
        logging.info('Saved figure %s' % self._out_cnm1_fig_name)
        if not self.showPlots:
            fig.set_visible(False)
            plt.close()
        self._update_progressbar('cnmf1')

        # next, eval component quality from stage 1
        # the components are evaluated in three ways:
        #   a) the shape of each component must be correlated with the data
        #   b) a minimum peak SNR is required over the length of a transient
        #   c) each shape passes a CNN based classifier

        cnm = self.cnm1  # shorthand for below
        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            (idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds) \
                = cm.components_evaluation.estimate_components_quality_auto(
                       images, cnm.A, cnm.C, cnm.b, cnm.f,
                       cnm.YrA, frPerS, decayTimeS, gSig, dims,
                       dview=dview, min_SNR=min_SNR,
                       r_values_min=rval_thr, use_cnn=use_cnn,
                       thresh_cnn_lowest=cnn_thr)

        # plots of accepted and rejected components
        fig = plt.figure(figsize=r_[1,0.75]*16)
        exargs = { 'number_args': {'fontsize': 8}, 'contour_args': {'linewidth': 0.25},
                   'thr_method':'nrg', 'thrnrg':0.9 }
        plt.subplot(121)
        crd_good = cm.utils.visualization.plot_contours(cnm.A[:, idx_components], Cn, vmax=0.75, **exargs)
        plt.title('Contour plots of accepted components')
        plt.subplot(122)
        crd_bad = cm.utils.visualization.plot_contours(cnm.A[:, idx_components_bad], Cn, vmax=0.75, **exargs)
        plt.title('Contour plots of rejected components')
        fig.savefig(self._out_cnm_rej_fig_name)
        logging.info('Saved figure %s' % self._out_cnm_rej_fig_name)
        if not self.showPlots:
            fig.set_visible(False)
            plt.close()

        # now run stage 2, rerun seeded CNMF on accepted patches to refine and perform deconvolution
        A_in, C_in, b_in, f_in = cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f
        self.cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                         merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                         f_in=f_in, rf=None, stride=None, gnb=gnb,
                         method_deconvolution='oasis', check_nan=True)
        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            self.cnm2 = self.cnm2.fit(images)
        self.cnm = self.cnm2   # master cnf is the last
        self._update_progressbar('cnmf2')

        # extract df/f, both ways
        logger.info('Starting df/f extraction')
        with contextlib.redirect_stdout(self._stdoutFh), contextlib.redirect_stderr(self._stdoutFh):
            self.F_dff = cm.source_extraction.cnmf.utilities.extract_DF_F(Yr, self.cnm2.A, self.cnm2.C, self.cnm2.bl,
                                                                 quantileMin=8, frames_window=200, block_size=400,
                                                                 dview=None)
            if self.F_dff.dtype == 'O':
                warnings.warn('object dtype found - check, may need fix??  Trying to convert to double/f8')
                self.F_dff = self.F_dff.astype('f8')


            try:
                # if we get exceptions here, we can catch them and just skip this
                self.F_dff_model = detrend_df_f(self.cnm2.A, self.cnm2.b, self.cnm2.C, self.cnm2.f, YrA=self.cnm2.YrA,
                                   quantileMin=5, frames_window=250)
            except ValueError as exc:
                logger.warning('F_dff_model failed, skipping')
                self.F_dff_model = []
        self._update_progressbar('df/f')




        # save outputs
        logger.debug('Saving output to %s' % self._out_cnm_data_name)
        cnm2 = self.cnm2 # alias shortcut
        # 180805: changed save mechanism to use joblib and save cnm2 obj specifically
        #np.savez(self._out_cnm_data_name,
        #         Cn=Cn, A=cnm2.A.todense(), C=cnm2.C,
        #         S=cnm2.S, F_dff=self.F_dff, F_dff_model=self.F_dff_model,
        #         b=cnm2.b, f=cnm2.f, YrA=cnm2.YrA, sn=cnm2.sn, d1=d1, d2=d2,
        #         frPerS=frPerS)
        # 190527: v3 save: drop the object as it gets pickled and requires caiman for load
        joblib.dump(Namespace(cnm=cnm2, localCorr=Cn,
                              F_dff=self.F_dff, F_dff_model=self.F_dff_model,
                              d1=d1, d2=d2, frPerS=frPerS),
                    self._out_cnm_data_name)
        # helper to make trace figure to make this function shorter
        self._cnmf_make_trace_figures()

        logger.info('CNMF done.')
        self._update_progressbar('cnmf_save')







    def _cnmf_make_trace_figures(self):
        """Make diagnostic figures from cnm output, save to disk
        Uses:
            self.cnm

        """
        cnm = self.cnm

        gs = mpl.gridspec.GridSpec(4, 1)
        fig = plt.figure(figsize=r_[1, 0.75] * 10)
        nCells, nFrs = cnm.S.shape

        xs = r_[:nFrs]
        xIx = xs < 200

        ax = plt.subplot(gs[0, 0])
        plt.plot(cnm.C[:, xIx].T)
        plt.title('C component - ARMA process')

        ax = plt.subplot(gs[1, 0])
        plt.plot(self.F_dff[:, xIx].T)
        plt.title('computed df/F traces')
        # plt.ylim([0,300])

        ax = plt.subplot(gs[2, 0])
        plt.plot(cnm.S[:, xIx].T)
        plt.title('S - estimated spikes')
        plt.xlabel('Frames (5 Hz)')

        ax = plt.subplot(gs[3, 0])
        r = np.mean(cnm.S, axis=0)
        smR = ptMH.math.smooth(r, span=30, method='lowess', robust=False)
        plt.plot(xs[xIx], r[xIx])
        plt.plot(xs[xIx], smR[xIx], lw=5)
        # smS = ss.savgol_filter(cnm2.S, window_length=91, polyorder=1, axis=1)
        # plt.plot(smS[:,xIx].T)
        # plt.plot(ss.savgol_filter(cnm2.S.mean(axis=0), window_length=91, polyorder=1))

        plt.title(r'$\bar{S}$ over cells')
        plt.xlabel('Frames (5 Hz)')

        plt.tight_layout()

        fig.set_size_inches(r_[1, 0.75] * [2, 4] * 6)
        fig.savefig(self._out_cnm_trace_fig_name)
        if not self.showPlots:
            fig.set_visible(False)
            plt.close()


###################################################
## util fns

def pctile_hist(m, artifactPct=2, doPlot=True):
    """Compute histogram of line top percentile pix.  2% is highest 5 pix
    Returns:
        lineTopPM: ndarray shape(nLines*nFr,), top percentile pix val of each line
        lineThresh: scalar, appropriate threshold for finding artifact lines.  Shown on plot as vert line
        """
    y0 = np.percentile(m, 100-artifactPct, axis=2).ravel()
    maxX = y0.max()*1.1
    lineThresh = y0.max()/2
    if doPlot:
        ax = plt.gca()
        ax.hist(y0, bins=np.linspace(0,maxX,500), histtype='step', color='r')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_xlabel('Line intensity, %d percentile'%(100-artifactPct))
        ax.set_xlim(0, maxX)
        ax.set_yticks([])
        ax.set_yscale('log')
        ax.axvline(lineThresh, lw=0.25, color='k')
    return(y0, lineThresh)
