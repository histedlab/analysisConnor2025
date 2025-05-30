import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tfl
from argparse import Namespace
import glob
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf import params
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.utils.visualization import plot_contours

import logging

logger = logging.getLogger(__name__)

# import custom packages
import pytoolsMH as ptMH

# define operations
a_ = np.asarray
r_ = np.r_


class CaimanRun:
    """Does a full caiman run, saving intermediate images and all outputs to our set locations.

    Inputs: The filename of imageFrames.tif, or its root directory.
    Outputs: analysisCaiman directory

    Call as a context manager/with statement:

    with CaimanRun(fname) as cr:
        cr.dostep1
        cr.dostep2

    This ensures cleanup happens properly.

    Attribs:
        workstackfullname: current temp tif, output of each manipulation step, deleted at end; full name with path
        origstackname: source file, imageFrames.tif, no path

    Notes:
        - We don't keep the movie tiff in memory, we read it as needed, since that's what Caiman works with: on disk files
        - Because we're using local computation for now, we start/stop clusters inside each method step.  If we later
          wish to run against a bigger longer running cluster, add separate methods to start/stop and test for running
          inside the methods
        - We define here as many output filenames and other constants as we can, so they are in one place
        - We save the mc output as a uint16 TIF file.  A few notes:
          - The motiin correction output can have negative pixel vals, esp with pw_rigid.
          - Fiji doesn't seem to be able to read tifffile's int16 tifs, so we subtract the min pixel val
            (giving min = 0 and no neg vals), round (instead of truncate) and write it as uint16.
          - Caiman writes a float32 mmap file as output from mc.  Even when we round first and then convert to 16-bit,
            the CNMF code appears to do a worse job on the tifs, so suggest always doing mc first when running CNMF.
            I didn't save as float32 as this doubles the file size.

    """

    def __init__(self, path, fname="imageFrames.tif", caiman_opts_dict={}):
        """
        Args: 
            path, fname to the stack
            caiman_opts_dict: see caiman docs
        """
        opts_ourdefaults_dict = {
            "fr": 30,
            "decay_time": 1.2,
            "strides": (48, 48),
            "overlaps": (24, 24),
            "max_shifts": (6, 6),
            "max_deviation_rigid": 3,
            "pw_rigid": False,
            "shifts_opencv": True,
            "border_nan": "min",
            "num_frames_split": 80,
            "nonneg_movie": True,
            "p": 1,
            "nb": 2,
            "merge_thr": 0.8,
            "rf": 15,
            "stride": 6,
            "K": 4,
            "gSig": [4, 4],
            "method_init": "greedy_roi",
            "min_SNR": 2.5,
            "rval_thr": 0.8,
            "use_cnn": False,
            "min_cnn_thr": 0.8,
            "cnn_lowest": 0.1,
            "min_mov": None,
        }
        self.basedir = path
        self._enter_called = False
        self.fname = [os.path.join(path, fname)]
        self.outdir = os.path.join(path, "analysisCaiman")
        os.makedirs(self.outdir, exist_ok=True)
        self.mc_data = None
        if not os.path.exists(self.fname[0]):
            raise RuntimeError("Cannot find stack: %s" % self.fname[0])

        ## output names
        # results
        self._out_results_mc_name = os.path.join(self.outdir, "results-mc.npz")
        self._out_results_analysis_npz = os.path.join(self.outdir, "results-analysis-v3.npz")

        # mmaps
        self._cnmf_mmap_name_base = os.path.join(self.basedir, "cnmf_mmap_")

        # tifs
        self._out_dropFrs_tif_name = os.path.join(self.basedir, "imageFrames-dropFrs.tif")
        self._out_mc_tif_name = os.path.join(self.basedir, "imageFrames-mc.tif")

        # figs
        self._out_blank_fig_name = os.path.join(self.outdir, "fig-blank-stats.png")  # too many objs for pdf
        self._out_cnm1_fig_name = os.path.join(self.outdir, "cnmf-run1-contours.pdf")
        self._out_cnm2_fig_name = os.path.join(self.outdir, "cnmf-run2-contours.pdf")
        self._out_cnmf1_fig_name = os.path.join(self.outdir, "cnmf-run1-eval.pdf")
        self._out_cnmf2_fig_name = os.path.join(self.outdir, "cnmf-run2-eval.pdf")
        self._out_cnm_trace_fig_name = os.path.join(self.outdir, "trace_examples.pdf")

        # for drop artifact frames
        self.frames_dropped = False
        self.keep_dropArtifact_tif = False

        # create params dictionary
        opts_ourdefaults_dict.update(caiman_opts_dict)
        opts_ourdefaults_dict["fnames"] = self.fname

        self.opts = params.CNMFParams(params_dict=opts_ourdefaults_dict)
        

        # some option checks here
        if not self.opts.get("motion", "min_mov") is None:
            logger.warning("min_mov may be able to be None: neg vals removed in tif save after mc")

    def __enter__(self):
        self._enter_called = True

        # start cluster (if one already started, try using the old one; faster)
        # if "dview" in locals():
        #    cm.stop_server(dview=dview)
        if not hasattr(self, "dview"):
            # Start is fast for local/multiprocessing.  If we ever want it to persist across objects,
            # say when using a slower dview like ipyparallel, we could persist in a global variable, etc.
            # note self.dview disappears when this context manager object is deleted.

            # logger.warning('Starting cluster')
            c, self.dview, self.dview_n_processes = cm.cluster.setup_cluster(
                backend="local", n_processes=None, single_thread=False
            )
            logger.warning("Started cluster.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
        self.dview.terminate()
        return

    def _cleanup(self):
        """Removes analysis related files from temporary directory"""
        # if motion corrected image stack came from imageFrames-dropArtifact.tif, then delete if needed
        if not self.keep_dropArtifact_tif and self.frames_dropped:
            os.remove(self.fname[0])
        tmpmmap = glob.glob(os.path.join(self.basedir, "cnmf*.mmap"))

        for tF in tmpmmap:
            os.remove(tF)

    def drop_artifact_frs(self, nPreStimFr, nPostStimFr, nArtifactFr=1, keeptif=False, plotResults=True):
        """For 1p stim imaging: drop frames that contain stim artifacts"""
        (nFr, nX, nY) = ptMH.image.tif_file_get_dims(self.fname[0])
        if nArtifactFr > 1:
            raise RuntimeError("Need to add to this code if more than one artifact frame")

        # drop frames and calculate statistics before and after
        dropFrs = r_[nPreStimFr : nFr : nPreStimFr + nPostStimFr]
        (origstats, newstats) = self._movie_drop_and_write_new(dropFrs, doStats=True)

        # set variable inidcate frames dropped and to keep or remove new image stack
        self.frames_dropped = True
        self.keep_dropArtifact_tif = keeptif

        # plot comparative statistics between original and new (dropped frames) image stacks
        if plotResults:
            fig = plt.figure(figsize=r_[1, 0.75] * 12)
            gs = mpl.gridspec.GridSpec(2, 2)

            # first row, all data
            ax = plt.subplot(gs[0, 0])
            plt.plot(origstats.framemean)
            plt.plot(newstats.framemean)
            ax = plt.subplot(gs[0, 1])
            plt.plot(origstats.linemean)
            plt.plot(newstats.linemean)

            # second row, zoomed data
            frR = [0, 200]
            lineR = [nY * (nPreStimFr - 1), nY * (nPreStimFr + 2)]
            ax = plt.subplot(gs[1, 0])
            plt.plot(origstats.framemean[frR[0] : frR[1]])
            plt.plot(newstats.framemean[frR[0] : frR[1]])
            plt.xlabel("frames")
            ax = plt.subplot(gs[1, 1])
            xs = origstats.linexs
            desNs = r_[lineR[0] : lineR[1]]
            plt.plot(xs[desNs], origstats.linemean[desNs])
            plt.plot(xs[desNs], newstats.linemean[desNs])
            linex = r_[lineR[0] : lineR[1] : nY]
            for x in linex:
                ax.axvline(x, lw=0.25)
            plt.xlabel("lines")

            # some adjs
            for ax in fig.get_axes():
                if ax.is_first_col():
                    ax.set_ylabel("pix val")

            # save figure
            plt.show()  # give us the artifact drop plots immediately so we can make sure they are right before mc/cnmf
            fig.savefig(self._out_blank_fig_name)  # lots of points, so save as png

    def _movie_compute_pixstats(self, m):
        """Helper function to compute some pixel statistics: frame mean, line mean, etc
        Notes:
            Can add more stats to output namespace as desired
        """
        (nFr, nY, nX) = m.shape
        return Namespace(
            framemean=m.mean(axis=2).mean(axis=1),
            linemean=m.mean(axis=2).reshape((nFr * nY)),
            framexs=r_[:nFr],
            linexs=r_[: nFr * nY],
        )

    def _movie_drop_and_write_new(self, dropFrs=None, keepFrs=None, checkMinGb=0.5, doStats=False):
        """Write a new movie after dropping (artifact?) frames.  Always uint16.
        Returns:
            None
        Sets:
            self.workstackfullname
            self._workframemean
            self._worklinemean
        """
        # define file name for dropped frame image stack
        drpFrs_fname = self._out_dropFrs_tif_name

        # load file;  copy frame after artifact frames into artifact frame
        m = cm.load(self.fname[0])
        nFr = m.shape[0]
        if keepFrs is not None:
            assert dropFrs is None, "cannot specify both dropFrs and keepFrs"
        else:
            # compute keepFrs from dropFrs
            keepFrs = np.setdiff1d(r_[0:nFr], dropFrs)
        # compute stats on original image stack
        origstats = self._movie_compute_pixstats(m)

        # remove artifact frames
        m = m[keepFrs, :, :]

        # save image stack
        tfl.imsave(drpFrs_fname, m.astype("uint16"))

        # set working image stack to newly created stack with dropped frames
        self.fname[0] = drpFrs_fname

        # compute stats on new image stack
        if doStats:
            finalstats = self._movie_compute_pixstats(m)
            return (origstats, finalstats)
        else:
            return None
        
    def remove_hpx_artifact_by_imputation(self, skipChecks=False):
        """odd artifact removal for lambda hpx, won't be used that often.

        Notes:
             don't update the progress bar for this one, too infrequent.
        """
        m = cm.load(self.fname[0])
        fig,(ax1,ax2) = plt.subplots(1,2)
        plt.sca(ax1)
        (pctLines, lineThresh) = self._pctile_hist(m,artifactPct=2)
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
        self._pctile_hist(m2, 2)
        plt.title('nPix imputed: %d, in %d lines' % (a_(pixLImputed).sum(), len(pixLImputed)))
        # now save
        tfl.imsave(self.fname[0], m2)
        out_hpxartifact_fig_name = os.path.join(self.outdir, 'fig-hpx-artifact-removal-stats.pdf')
        fig.savefig(out_hpxartifact_fig_name)
        plt.show()
    
    def _pctile_hist(self, m, artifactPct=2, doPlot=True):
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

    def run_motion_correction(self, save_mc_as_tif=False, mc_template=None):
        """
        Run motion correction on a local cluster
        """
        logger.warning("Starting motion correction: pw_rigid %s" % self.opts.motion["pw_rigid"])
        # compute the motion corrected image stack
        self.mc = MotionCorrect(self.fname, dview=self.dview, **self.opts.get_group("motion"))
        self.mc.motion_correct(save_movie=True, template=mc_template)

        # set parameters based on piecewise or rigid motion correction
        if self.opts.get("motion", "pw_rigid"):
            self._mc_mmap_name = self.mc.fname_tot_els
            self._bord_px = np.ceil(
                np.maximum(np.max(np.abs(self.mc.x_shifts_els)), np.max(np.abs(self.mc.y_shifts_els)))
            ).astype(int)
        else:
            self._mc_mmap_name = self.mc.fname_tot_rig
            self._bord_px = np.ceil(
                np.maximum(np.max(np.abs(self.mc.shifts_rig)), np.max(np.abs(self.mc.shifts_rig)))
            ).astype(int)
            
        #####################
        # create outD dictionary of parameters from mc object (note, not self object)
        outD = {}
        dropNL = ["dview"]  # manually drop these fields
        for tN in dir(self.mc):
            tV = getattr(self.mc, tN)
            if tN[0:2] == "__" or callable(tV) or tN in dropNL:  # drop methods or dunder atttribs
                continue
            else:
                outD[tN] = tV
        outD["bord_px"] = self._bord_px

        # save parameter dictionary
        np.savez(self._out_results_mc_name, **outD)
        self.mc_data = Namespace(**outD)
        #####################    
    
        # save motion corrected movie as tif if requested
        if save_mc_as_tif:
            mc_mmap = cm.load(self._mc_mmap_name)
            min_mc_pixval = np.min(mc_mmap)
            if min_mc_pixval != 0:
                logger.warning("While saving mc output tif: subtr min pix val %d" % min_mc_pixval)

            tfl.imsave(self._out_mc_tif_name, np.around(mc_mmap - min_mc_pixval).astype("uint16"))

        # assign the file that cnmf will read for next step
        self._cnmf_readL = self._mc_mmap_name  # this is already a list
        assert (
            type(self._cnmf_readL) == list
        ), "this should already be a list"  # otherwise error in cm.save_memmap

    def load_motion_correct_mmap(self):
        """Mmap is saved as float32 (Which is what mc generates) and so CNMF may do better with it.
        Sets:
            self._cnmf_readL
            self._bord_px
        """
        self.mc_data = Namespace(**dict(np.load(self._out_results_mc_name, allow_pickle=True)))
        self._cnmf_readL = [str(self.mc_data.mmap_file[0])]  # npz save changes type to ndarray string type
        self._bord_px = self.mc_data.bord_px

    def load_motion_correct_tif(self, force_file_name=None, bord_px=None):
        """Load previously run motion correction tif.
        Args:
            bord_px: if None, try to read results file and get value from there.  Otherwise,
            use this value as bord_px and don't read the mc npz file.
        Sets:
            self._cnmf_readL
            self._bord_px
        Notes:
            The tif is saved as uint16 and so may lose some info relative to the mmap file
            Requires save_mc_as_tif=True
        """
        logger.warning("Reading motion correction uint16 data from tif.  NOT as accurate as mmap(float32).")

        if force_file_name is not None:
            readf = force_file_name
        else:
            readf = self._out_mc_tif_name
        self._cnmf_readL = [readf]
        if bord_px is None:
            self.mc_data = Namespace(**dict(np.load(self._out_results_mc_name, allow_pickle=True)))
            self._bord_px = self.mc_data.bord_px
        else:
            self._bord_px = bord_px


    def run_cnmf_twostep(self, extract_dff=False, cnmfRuns=1):
        """
        Reads attribs:
            self._cnmf_readL
            self._bord_px

        Run CNMF two step:
        1) Fit CNMF and evaluate components
        2) Refit first CNMF and re-evaluate components + extract source
        """
        logger.warning("Starting CNMF: cnmfRuns=%d" % cnmfRuns)

        if not os.path.exists(self._cnmf_readL[0]):
            raise RuntimeError("input stack for cnmf: mmap/tiff file- not found: %s" % self._cnmf_readL[0])

        self.extract_dff = extract_dff

        # load motion correction mmap and save in C order
        self._cnmf_mmap_name = cm.save_memmap(
            self._cnmf_readL, base_name=self._cnmf_mmap_name_base, order="C", border_to_0=self._bord_px
        )

        # load C order mmap for cnmf
        Yr, dims, T = cm.load_memmap(self._cnmf_mmap_name)
        self.d1, self.d2 = dims

        images = np.reshape(Yr.T, [T] + list(dims), order="F")

        ## change p to 0 (default 1)
        # if cnmfRuns == 2:
        #    self.opts.change_params({"p": 0})

        # instantiate CNMF object and do first fit
        self.cnm = cnmf.CNMF(self.dview_n_processes, params=self.opts, dview=self.dview)
        #self.cnm.params.change_params({'nb':0})


        self.cnm = self.cnm.fit(images)

        logger.warning('Done with CNMF.')

        # refit with seed from last
        #doRefit = True
        doRefit = False
        if doRefit:
            cnm1 = self.cnm
            self.cnm = self.cnm.refit(images, dview=self.dview)
            logger.warning('Done with CNMF fit 2')

        # define contours variable
        Cn = cm.local_correlations(images.transpose(1, 2, 0))
        Cn[np.isnan(Cn)] = 0
        self.Cn = Cn

        # set estimates object
        cnme = self.cnm.estimates

        # evaluate components
        (
            idx_components,
            idx_components_bad,
            SNR_comp,
            r_values,
            cnn_preds,
        ) = estimate_components_quality_auto(
            images,
            cnme.A,
            cnme.C,
            cnme.b,
            cnme.f,
            cnme.YrA,
            frate=self.opts.get("data", "fr"),
            decay_time=self.opts.get("data", "decay_time"),
            gSig=self.opts.get("init", "gSig"),
            dims=self.opts.get("data", "dims"),
            dview=self.dview,
            min_SNR=self.opts.get("quality", "min_SNR"),  # fn def 2.0
            r_values_min=self.opts.get("quality", "rval_thr"),  # fn def 0.9
            r_values_lowest=self.opts.get("quality", "rval_lowest"),  # fn def -1, us -1
            use_cnn=self.opts.get("quality", "use_cnn"),
            thresh_cnn_min=self.opts.get("quality", "min_cnn_thr"),  # fn def 0.95, us 0.8
            thresh_cnn_lowest=self.opts.get("quality", "cnn_lowest"),  # fn def 0.1, our def 0.1
        )
        # create alias for evaluation results
        self.idx_accept = idx_components
        self.idx_reject = idx_components_bad
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds

        # plot figure of all found components
        fig = plt.figure()
        crd = plot_contours(self.cnm.estimates.A, Cn, thr_method="nrg", thrnrg=0.9, fontsize=9)
        plt.title("Contour plots of found components")
        fig.set_size_inches(r_[1, 0.75] * 16)
        plt.savefig(self._out_cnm1_fig_name)
        # plot figure of accepted and rejected components (run 1)
        fig = plt.figure(figsize=r_[1, 0.75] * 16)
        exargs = {
            "number_args": {"fontsize": 8},
            "contour_args": {"linewidth": 0.25},
            "thr_method": "nrg",
            "thrnrg": 0.9,
        }
        plt.subplot(121)
        crd_good = cm.utils.visualization.plot_contours(cnme.A[:, idx_components], Cn, vmax=0.75, **exargs)
        plt.title("Contour plots of accepted components")
        plt.subplot(122)
        crd_bad = cm.utils.visualization.plot_contours(cnme.A[:, idx_components_bad], Cn, vmax=0.75, **exargs)
        plt.title("Contour plots of rejected components")
        plt.savefig(self._out_cnmf1_fig_name)
        print("# components after cnm fit 1: ", cnme.C.shape[0])
        print("# accepted components from auto eval after cnm fit 1: ", idx_components.shape[0])

        if cnmfRuns == 2:
            # now run stage 2, rerun seeded CNMF on accepted patches to refine and perform deconvolution
            self.opts.change_params({"p": 1})

            # now run stage 2, rerun seeded CNMF on accepted patches to refine and perform deconvolution
            A_in, C_in, b_in, f_in = (cnme.A[:, idx_components], cnme.C[idx_components], cnme.b, cnme.f)
            cnm2 = cnmf.CNMF(
                n_processes=self.dview.n_processes,  # note 2018 caiman didn't work with n_processes > 1, this appears fine
                k=A_in.shape[-1],
                Ain=A_in,
                Cin=C_in,
                b_in=b_in,
                f_in=f_in,
                gSig=self.gSig,
                gnb=self.gnb,
                rf=None,
                stride=None,
                p=1,
                method_deconvolution="oasis",
                check_nan=True,
                dview=self.dview,
            )
            self.cnm2 = cnm2.fit(images)
            #         cnm2 = self.cnm.refit(images, dview=dview)
            cnm2e = self.cnm2.estimates

            # evaluate the components
            (
                idx_components,
                idx_components_bad,
                SNR_comp,
                r_values,
                cnn_preds,
            ) = estimate_components_quality_auto(
                images,
                cnm2e.A,
                cnm2e.C,
                cnm2e.b,
                cnm2e.f,
                cnm2e.YrA,
                self.fps,
                self.decay_time,
                self.gSig,
                dims,
                dview=self.dview,
                min_SNR=self.min_SNR,
                r_values_min=self.rval_thr,
                use_cnn=self.use_cnn,
                thresh_cnn_lowest=self.cnn_thr,
            )

            # create alias for evaluation results
            self.idx_accept = idx_components
            self.idx_reject = idx_components_bad
            self.SNR_comp = SNR_comp
            self.r_values = r_values
            self.cnn_preds = cnn_preds

            # plot figure of all found components
            fig = plt.figure()
            crd = plot_contours(cnm2.estimates.A, Cn, thr_method="nrg", thrnrg=0.9, fontsize=9)
            plt.title("Contour plots of found components")
            fig.set_size_inches(r_[1, 0.75] * 16)
            plt.savefig(self._out_cnm2_fig_name)
            # plot figure of devoncolved, refined components
            fig = plt.figure(figsize=r_[1, 0.75] * 16)
            exargs = {
                "number_args": {"fontsize": 8},
                "contour_args": {"linewidth": 0.25},
                "thr_method": "nrg",
                "thrnrg": 0.9,
            }
            plt.subplot(121)
            crd_good = cm.utils.visualization.plot_contours(
                cnm2e.A[:, idx_components], Cn, vmax=0.75, **exargs
            )
            plt.title("Contour plots of accepted components")
            plt.subplot(122)
            crd_bad = cm.utils.visualization.plot_contours(
                cnm2e.A[:, idx_components_bad], Cn, vmax=0.75, **exargs
            )
            plt.title("Contour plots of rejected components")
            plt.savefig(self._out_cnmf2_fig_name)
            print("after cnm refit 2: ", cnme.C.shape)
            print("from auto eval after cnm refit 2: ", idx_components.shape)

        # save output
        self._save_final_output(cnmfRuns=cnmfRuns)

        # plot cell traces of various data
        self._cnmf_make_trace_figures(cnmfRuns=cnmfRuns)

    def _save_final_output(self, cnmfRuns):
        idx_accept = self.idx_accept
        if cnmfRuns == 1:
            cnme = self.cnm.estimates
        elif cnmfRuns == 2:
            cnme = self.cnm2.estimates
        # grab only accepted components
        A_out = cnme.A[:, idx_accept]
        C_out = cnme.C[idx_accept, :]
        S_out = cnme.S[idx_accept, :]
        YrA_out = cnme.YrA[idx_accept, :]

        # create dictionary to store output in
        results_analysis = {
            "A": A_out,
            "C": C_out,
            "S": S_out,
            "YrA": YrA_out,
            "b": cnme.b,
            "f": cnme.f,
            "opts": self.opts.to_dict(),
            "Cn": self.Cn,
            "d1": self.d1,
            "d2": self.d2,
            "frPerS": self.opts.data["fr"],
        }
        if not hasattr(cnme, 'Breg') or cnme.Breg is None:
            results_analysis['Breg'] = None
        else:
            results_analysis['Breg'] = cnme.Breg[idx_accept,:]
            print(results_analysis['Breg'])

        np.savez(self._out_results_analysis_npz, results_dict=results_analysis)

    def _cnmf_make_trace_figures(self, cnmfRuns=1):
        """
        Make diagnostic figures from cnm output, save to disk
        Uses:
            self.cnm
        """
        cnme = self.cnm.estimates
        idx_accept = self.idx_accept
        if cnmfRuns == 1:
            cnme = self.cnm.estimates
        elif cnmfRuns == 2:
            cnme = self.cnm2.estimates
        # grab only accepted components
        C = cnme.C[idx_accept, :]
        S = cnme.S[idx_accept, :]

        # add plot row for dff if extracted
        if self.extract_dff:
            plot_rows = 4
        else:
            plot_rows = 3

        gs = mpl.gridspec.GridSpec(plot_rows, 1)
        fig = plt.figure(figsize=r_[1, 0.75] * 10)
        nCells, nFrs = S.shape

        xs = r_[:nFrs]
        xIx = xs < 200

        ax = plt.subplot(gs[0, 0])
        plt.plot(C[:, xIx].T)
        plt.title("C component - ARMA process")

        ax = plt.subplot(gs[1, 0])
        plt.plot(S[:, xIx].T)
        plt.title("S - estimated spikes")
        plt.xlabel("Frames (5 Hz)")

        ax = plt.subplot(gs[2, 0])
        r = np.mean(S, axis=0)
        plt.plot(xs[xIx], r[xIx])

        # pytoolsMH plotting module won't load because scipy.misc.factorial is deprecated and is called by statsmodels package
        # keeping commented out until later release of statsmodel is available (soon?)
        # see https://github.com/statsmodels/statsmodels/issues/5747 for temporary fix (downgrade scipy or install from master)
        # and https://github.com/statsmodels/statsmodels/issues/5620 for next release discussion of statsmodels
        #         smR = ptMH.math.smooth(r, span=30, method='lowess', robust=False)
        #         plt.plot(xs[xIx], smR[xIx], lw=5)

        plt.title(r"$\bar{S}$ over cells")
        plt.xlabel("Frames (5 Hz)")

        # plot dff if extracted
        if self.extract_dff:
            ax = plt.subplot(gs[3, 0])
            plt.plot(self.F_df[:, xIx].T)
            plt.title("computed df/F traces")

        fig.set_size_inches(r_[1, 0.75] * [2, plot_rows + 1] * 6)
        plt.savefig(self._out_cnm_trace_fig_name)
