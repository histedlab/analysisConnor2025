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

class CaimanRunHD():
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

    Todo:
        - We currently enumerate a lot of options in params to init().  Caiman now uses an 'opts' object to list all the
          options and classify them into groups.  We should probably just use a **dict here to set all the options we
          we want and save a lot of code.  MH 190710

    """

    def __init__(self, path, fname='imageFrames.tif', fps=5, decay_time=1.2, 
                 strides=(48, 48), template=None,
                 overlaps=(24,24), max_shifts=(6,6), max_deviation_rigid=3,
                 pw_rigid=False, shifts_opencv=True, border_nan='copy',
                 num_frames_split=80, nonneg_movie=True, p=1, gnb=2,
                 merge_thr=0.8, rf=15, stride_cnmf=6, K=4, gSig=[3,3],
                 method_init='greedy_roi', min_SNR=2.5, rval_thr=0.8, 
                 use_cnn=False, cnn_thr=0.8, cnn_lowest=0.1,
                 min_mov=None,
                 ext_opts={}):
        """
        Args: 
            path, fname to the stack
            Motion correction params - see Caiman docs
            CNMF params - see Caiman docs
        """
        self.basedir = path
        self.fname = [os.path.join(path,fname)]
        self.fps = fps
        self.decay_time = decay_time
        self.pw_rigid = pw_rigid
        self.gSig = gSig
        self.gnb = gnb
        self.min_SNR = min_SNR
        self.rval_thr = rval_thr
        self.use_cnn = use_cnn
        self.cnn_thr = cnn_thr
        self.min_mov = min_mov
        self.template = template
        
        self.outdir = os.path.join(path,'analysisCaiman')
        os.makedirs(self.outdir, exist_ok=True)
        
        self._enter_called = False
        self.mc_data = None
        if not os.path.exists(self.fname[0]):
            raise RuntimeError('Cannot find stack: %s' % self.fname[0])

        ## output names
        # results
        self._out_results_mc_name = os.path.join(self.outdir,'results-mc.npz')
        self._out_results_analysis_npz = os.path.join(self.outdir,'results-analysis-v3.npz')
        
        # mmaps
        self._cnmf_mmap_name_base = os.path.join(self.basedir, 'cnmf_mmap_')
        
        # tifs
        self._out_dropFrs_tif_name = os.path.join(self.basedir,'imageFrames-dropFrs.tif')
        self._out_mc_tif_name = os.path.join(self.basedir,'imageFrames-mc.tif')
        
        # figs
        self._out_blank_fig_name = os.path.join(self.outdir, 'fig-blank-stats.png') # too many objs for pdf
        self._out_cnm1_fig_name = os.path.join(self.outdir,'cnmf-run1-contours.pdf')
        self._out_cnm2_fig_name = os.path.join(self.outdir,'cnmf-run2-contours.pdf')
        self._out_cnmf1_fig_name = os.path.join(self.outdir,'cnmf-run1-eval.pdf')
        self._out_cnmf2_fig_name = os.path.join(self.outdir,'cnmf-run2-eval.pdf')
        self._out_cnm_trace_fig_name = os.path.join(self.outdir, 'trace_examples.pdf')
        
        # for drop artifact frames
        self.frames_dropped = False
        self.keep_dropArtifact_tif = False
        
        # create params dictionary
        opts_dict = {'fnames': self.fname,
             'fr': fps,
             'decay_time': decay_time,
             'strides': strides,
             'overlaps': overlaps,
             'max_shifts': max_shifts,
             'max_deviation_rigid': max_deviation_rigid,
             'pw_rigid': pw_rigid,
             'shifts_opencv': shifts_opencv,
             'border_nan': border_nan,
             'num_frames_split': num_frames_split,
             'nonneg_movie': nonneg_movie,
             'p': p,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'only_init': True,
             'merge_thr': merge_thr,
             'min_SNR': min_SNR,
             'rval_thr': rval_thr,
             'use_cnn': use_cnn,
             'min_cnn_thr': cnn_thr,
             'cnn_lowest': cnn_lowest,
             'min_mov': min_mov,
            }
        opts_dict.update(ext_opts)
        
        self.opts = params.CNMFParams(params_dict=opts_dict)
        if not self.opts.get('motion','min_mov') is None:
            logger.error('min_mov intended to be None: negative values are removed after motion correction below')



    def __enter__(self):
        self._enter_called = True
        return self

    
    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()
        return
        
        
    def _cleanup(self):
        '''Removes analysis related files from temporary directory'''
        # if motion corrected image stack came from imageFrames-dropArtifact.tif, then delete if needed
        if not self.keep_dropArtifact_tif and self.frames_dropped:
            os.remove(self.fname[0])
        tmpmmap = glob.glob(os.path.join(self.basedir, 'cnmf*.mmap'))
        for tF in tmpmmap:
            os.remove(tF)

                
    def drop_artifact_frs(self, nPreStimFr, nPostStimFr, nArtifactFr=1, keeptif=False, plotResults=True):
        """For 1p stim imaging: drop frames that contain stim artifacts"""
        (nFr, nX, nY) = ptMH.image.tif_file_get_dims(self.fname[0])
        if nArtifactFr > 1:
            raise RuntimeError('Need to add to this code if more than one artifact frame')
       
        # drop frames and calculate statistics before and after
        dropFrs = r_[nPreStimFr:nFr:nPreStimFr + nPostStimFr]
        (origstats,newstats) = self._movie_drop_and_write_new(dropFrs, doStats=True)
        
        # set variable inidcate frames dropped and to keep or remove new image stack
        self.frames_dropped = True
        self.keep_dropArtifact_tif = keeptif
        
        # plot comparative statistics between original and new (dropped frames) image stacks
        if plotResults:
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

            # save figure
            fig.savefig(self._out_blank_fig_name)  # lots of points, so save as png
    

    def _movie_compute_pixstats(self, m):
        """Helper function to compute some pixel statistics: frame mean, line mean, etc
        Notes:
            Can add more stats to output namespace as desired
        """
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
        # define file name for dropped frame image stack
        drpFrs_fname = self._out_dropFrs_tif_name

        # load file;  copy frame after artifact frames into artifact frame
        m = cm.load(self.fname[0])
        nFr = m.shape[0]
        if keepFrs is not None:
            assert (dropFrs is None), 'cannot specify both dropFrs and keepFrs'
        else:
            # compute keepFrs from dropFrs
            keepFrs = np.setdiff1d(r_[0:nFr], dropFrs)
        # compute stats on original image stack
        origstats = self._movie_compute_pixstats(m)

        # remove artifact frames
        m = m[keepFrs,:,:]
        
        # save image stack 
        tfl.imsave(drpFrs_fname, m.astype('uint16'))
        
        # set working image stack to newly created stack with dropped frames
        self.fname[0] = drpFrs_fname

        # compute stats on new image stack
        if doStats:
            finalstats = self._movie_compute_pixstats(m)
            return (origstats, finalstats)
        else:
            return None
        
        
    def run_motion_correction(self,save_mc_as_tif=False):
        '''
        Run motion correction on a local cluster
        '''
        # start cluster (if one already started, stops and restarts)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
            
        # compute the motion corrected image stack
        self.mc = MotionCorrect(self.fname, dview=dview,**self.opts.get_group('motion'))
        self.mc.motion_correct(save_movie=True, template = self.template)
        
        # set parameters based on piecewise or rigid motion correction
        if self.pw_rigid:
            self._mc_mmap_name = self.mc.fname_tot_els
            self._bord_px = np.ceil(np.maximum(np.max(np.abs(self.mc.x_shifts_els)),\
                                    np.max(np.abs(self.mc.y_shifts_els)))).astype(np.int)
        else:
            self._mc_mmap_name = self.mc.fname_tot_rig
            self._bord_px = np.ceil(np.maximum(np.max(np.abs(self.mc.shifts_rig)),\
                                    np.max(np.abs(self.mc.shifts_rig)))).astype(np.int)

        # save motion corrected movie
        if save_mc_as_tif:
            mc_mmap = cm.load(self._mc_mmap_name)
            negIx = mc_mmap < 0
            nNeg = np.sum(negIx)
            if nNeg > 0:
                logger.warning('Removing %d (%.2g%%) negative values in movie (min val %.2g)'
                               % (nNeg, nNeg/np.prod(np.shape(mc_mmap)), np.min(mc_mmap[negIx])))
            #mc_mmap[negIx] = 0

            tfl.imsave(self._out_mc_tif_name, mc_mmap.astype('int16'))

        # create outD dictionary of parameters (not needed?)
        outD = {}
        dropNL = ['dview']  # manually drop these fields
        for tN in dir(self.mc):
            tV = getattr(self.mc, tN)
            if tN[0:2] == '__' or callable(tV) or tN in dropNL:  # drop methods or dunder atttribs
                continue
            else:
                outD[tN] = tV
        outD['bord_px'] = self._bord_px
        
        # save parameter dictionary
        np.savez(self._out_results_mc_name, **outD)
        self.mc_data = Namespace(**outD)

        # stop the server 
        cm.stop_server(dview=dview)
        
        
    def load_motion_correct_output(self):
        '''
        Load previously run motion correction output.
        Note: requires save_mc_as_tif=True
        '''
        self._mc_mmap_name = glob.glob(os.path.join(self.basedir, '*.mmap'))
        self.mc_data = Namespace(**dict(np.load(self._out_results_mc_name,allow_pickle=True))) # needed?
        self._bord_px = self.mc_data.bord_px
        
        
    def run_cnmf_twostep(self,extract_dff=False,quantileMin=8,frames_window=200,cnmfRuns=1):
        '''
        Run CNMF two step:
        1) Fit CNMF and evaluate components
        2) Refit first CNMF and re-evaluate components + extract source
        '''
        # set extract dff
        self.extract_dff = extract_dff
        
        # load motion correction mmap and save in C order
        self._cnmf_mmap_name = cm.save_memmap(self._mc_mmap_name, base_name=self._cnmf_mmap_name_base,\
                                              order='C',border_to_0=self._bord_px)
        
        # load C order mmap for cnmf
        Yr, dims, T = cm.load_memmap(self._cnmf_mmap_name)
        self.d1, self.d2 = dims

        images = np.reshape(Yr.T, [T] + list(dims), order='F') 
        
        # start local cluster
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

        # change p to 0 (default 1)
        if cnmfRuns == 2:
            self.opts.change_params({'p': 0})
        
        # instantiate CNMF object and do first fit
        self.cnm = cnmf.CNMF(n_processes, params=self.opts, dview=dview)
        self.cnm = self.cnm.fit(images)
    
        # define contours variable
        Cn = cm.local_correlations(images.transpose(1, 2, 0))
        Cn[np.isnan(Cn)] = 0
        self.Cn = Cn
        
        # set estimates object
        cnme = self.cnm.estimates
        
        # evaluate components
        (idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds)\
        = estimate_components_quality_auto(images, cnme.A, cnme.C, cnme.b, 
                                           cnme.f, cnme.YrA, self.fps, self.decay_time,
                                           self.gSig, dims, dview=dview, min_SNR=self.min_SNR,
                                           r_values_min=self.rval_thr, use_cnn=self.use_cnn, 
                                           thresh_cnn_lowest=self.cnn_thr)
        # create alias for evaluation results
        self.idx_accept = idx_components
        self.idx_reject = idx_components_bad
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds
        
        # plot figure of all found components  
        fig = plt.figure()
        crd = plot_contours(self.cnm.estimates.A, Cn, thr_method='nrg', thrnrg=0.9, fontsize=9)
        plt.title('Contour plots of found components')
        fig.set_size_inches(r_[1, 0.75] * 16)
        plt.savefig(self._out_cnm1_fig_name)
        # plot figure of accepted and rejected components (run 1)
        fig = plt.figure(figsize=r_[1,0.75]*16)
        exargs = { 'number_args': {'fontsize': 8}, 'contour_args': {'linewidth': 0.25},
                   'thr_method':'nrg', 'thrnrg':0.9 }
        plt.subplot(121)
        crd_good = cm.utils.visualization.plot_contours(
            cnme.A[:,idx_components], Cn, vmax=0.75, **exargs)
        plt.title('Contour plots of accepted components')
        plt.subplot(122)
        crd_bad = cm.utils.visualization.plot_contours(
            cnme.A[:,idx_components_bad], Cn, vmax=0.75, **exargs)
        plt.title('Contour plots of rejected components')
        plt.savefig(self._out_cnmf1_fig_name)
        print('# components after cnm fit 1: ', cnme.C.shape[0])
        print('# accepted components from auto eval after cnm fit 1: ',idx_components.shape[0])
        
        if cnmfRuns == 2:
            # now run stage 2, rerun seeded CNMF on accepted patches to refine and perform deconvolution
            self.opts.change_params({'p': 1})

            # now run stage 2, rerun seeded CNMF on accepted patches to refine and perform deconvolution
            A_in, C_in, b_in, f_in = cnme.A[:, idx_components], cnme.C[idx_components], cnme.b, cnme.f
            cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], Ain=A_in, Cin=C_in, b_in=b_in, f_in=f_in,
                                  gSig=self.gSig, gnb=self.gnb,
                                  rf=None, stride=None, p=1, method_deconvolution='oasis', check_nan=True, dview=dview)
            self.cnm2 = cnm2.fit(images)
    #         cnm2 = self.cnm.refit(images, dview=dview)
            cnm2e = self.cnm2.estimates

            # evaluate the components
            (idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds)\
            = estimate_components_quality_auto(images, cnm2e.A, cnm2e.C, cnm2e.b, 
                                               cnm2e.f, cnm2e.YrA, self.fps, self.decay_time,
                                               self.gSig, dims, dview=dview, min_SNR=self.min_SNR,
                                               r_values_min=self.rval_thr, use_cnn=self.use_cnn, 
                                               thresh_cnn_lowest=self.cnn_thr)
            
            # create alias for evaluation results
            self.idx_accept = idx_components
            self.idx_reject = idx_components_bad
            self.SNR_comp = SNR_comp
            self.r_values = r_values
            self.cnn_preds = cnn_preds

            # plot figure of all found components    
            fig = plt.figure()
            crd = plot_contours(cnm2.estimates.A, Cn, thr_method='nrg', thrnrg=0.9, fontsize=9)
            plt.title('Contour plots of found components')
            fig.set_size_inches(r_[1, 0.75] * 16)
            plt.savefig(self._out_cnm2_fig_name)
            # plot figure of devoncolved, refined components
            fig = plt.figure(figsize=r_[1,0.75]*16)
            exargs = { 'number_args': {'fontsize': 8}, 'contour_args': {'linewidth': 0.25},
                       'thr_method':'nrg', 'thrnrg':0.9 }
            plt.subplot(121)
            crd_good = cm.utils.visualization.plot_contours(
                cnm2e.A[:,idx_components], Cn, vmax=0.75, **exargs)
            plt.title('Contour plots of accepted components')
            plt.subplot(122)
            crd_bad = cm.utils.visualization.plot_contours(
                cnm2e.A[:,idx_components_bad], Cn, vmax=0.75, **exargs)
            plt.title('Contour plots of rejected components')
            plt.savefig(self._out_cnmf2_fig_name)
            print('after cnm refit 2: ', cnme.C.shape)
            print('from auto eval after cnm refit 2: ',idx_components.shape)
            
        # stop the server
        cm.stop_server(dview=dview)
        
        # save output
        self._save_final_output(cnmfRuns=cnmfRuns)
        
        # plot cell traces of various data
        self._cnmf_make_trace_figures(cnmfRuns=cnmfRuns)
        
        
    def _save_final_output(self,cnmfRuns):
        idx_accept = self.idx_accept
        if cnmfRuns == 1:
            cnme = self.cnm.estimates
        elif cnmfRuns == 2:
            cnme = self.cnm2.estimates
        # grab only accepted components
        A_out = cnme.A[:,idx_accept]
        C_out= cnme.C[idx_accept,:]
        S_out = cnme.S[idx_accept,:]
        YrA_out = cnme.YrA[idx_accept,:]
        
        # create dictionary to store output in
        results_analysis = {
            'A': A_out,
            'C': C_out,
            'S': S_out,
            'YrA': YrA_out,
            'b': cnme.b,
            'f': cnme.f,
            'Cn': self.Cn,
            'd1': self.d1,
            'd2': self.d2,
            'frPerS': self.fps
        }
        np.savez(self._out_results_analysis_npz,results_dict = results_analysis)
        
        
    def _cnmf_make_trace_figures(self,cells,cnmfRuns=1):
        '''
        Make diagnostic figures from cnm output, save to disk
        Uses:
            self.cnm
        '''
        cnme = self.cnm.estimates
        idx_accept = self.idx_accept
        if cnmfRuns == 1:
            cnme = self.cnm.estimates
        elif cnmfRuns == 2:
            cnme = self.cnm2.estimates
        # grab only accepted components
        C = cnme.C[idx_accept,:]
        S = cnme.S[idx_accept,:]
        
        # add plot row for dff if extracted
        if self.extract_dff:
            plot_rows = 4
        else:
            plot_rows = len(cells)

        #gs = mpl.gridspec.GridSpec(plot_rows, 1)
        #fig = plt.figure(figsize=r_[1, 0.75] * 10)
        nCells, nFrs = S.shape
        blocklen = 27000 / 3 #fix this to use actual movie length

        xs = r_[:nFrs]
        xIx = xs < (nFrs/2) #hardcoding until we can figure out why nFrs is doubled here

        
        fig, axs = plt.subplots(plot_rows, 1,sharex=True, sharey=True, gridspec_kw={'hspace': 0} )
        
        for cell_index in range(len(cells)):
            
            
            #ax = plt.subplot(gs[0, 0])
            #plt.plot(C[cell, xIx].T)
            #plt.title('C component - ARMA process')

            #ax = plt.subplot(gs[cell_index, 0])
            #plt.plot(S[cells[cell_index], xIx].T)
            #plt.title('S - estimated spikes')
            #plt.xlabel('Frames (5 Hz)')

            axs[cell_index].plot(S[cells[cell_index], xIx].T)
            axs[cell_index].set_ylim([0,20000])
            #axs[cell_index].plt.vlines(x=blocklen, ymin=0, ymax = 30000,color='red', zorder=2)
            #axs[cell_index].plt.vlines(x=blocklen*2, ymin=0,ymax = 30000, color='red', zorder=2)
            axs[cell_index].axvline(x= blocklen , color = 'red', ls = "--")
            axs[cell_index].axvline(x= blocklen*2 , color = 'red', ls = "--")

            #ax = plt.subplot(gs[2, 0])
            #r = np.mean(S, axis=0)
            #plt.plot(xs[xIx], r[xIx])
        
        # pytoolsMH plotting module won't load because scipy.misc.factorial is deprecated and is called by statsmodels package
        # keeping commented out until later release of statsmodel is available (soon?)
        # see https://github.com/statsmodels/statsmodels/issues/5747 for temporary fix (downgrade scipy or install from master)
        # and https://github.com/statsmodels/statsmodels/issues/5620 for next release discussion of statsmodels
#         smR = ptMH.math.smooth(r, span=30, method='lowess', robust=False)
#         plt.plot(xs[xIx], smR[xIx], lw=5)

        #plt.title(r'$\bar{S}$ over cells')
        #plt.xlabel('Frames (5 Hz)')
        for ax in axs:
            ax.label_outer()

        temp = 0
        for ax in axs.flat:
            ax.set(xlabel = "Frames (Hz)", ylabel="Cell {} \n Total Activity / min".format(cells[temp]))
            temp += 1

            
        # plot dff if extracted
        if self.extract_dff:
            ax = plt.subplot(gs[3, 0])
            plt.plot(self.F_df[:, xIx].T)
            plt.title('computed df/F traces')

        fig.set_size_inches(r_[1, 0.75] * [2, plot_rows+1] * 6)
        plt.savefig(self._out_cnm_trace_fig_name)
        plt.savefig('all_200-216.pdf')
        
    def _select_cells(self,percentile = 0.9,cnmfRuns=1):
        cnme = self.cnm.estimates
        idx_accept = self.idx_accept
        if cnmfRuns == 1:
            cnme = self.cnm.estimates
        elif cnmfRuns == 2:
            cnme = self.cnm2.estimates
        # grab only accepted components
        C = cnme.C[idx_accept,:]
        S = cnme.S[idx_accept,:]
        return (S)

