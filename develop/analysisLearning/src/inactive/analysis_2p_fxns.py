# ##### TO-DO:

# - within analysis_2p_general class
#     - generate_cell_masks_suite2p() fxn
#     - save image of aip with masks superimposed? (indicate stim vs. no-stim and number)
# - utils fxns
#     - load suite2p output
# - plotting fxns section???
#     - make a function to plot cell contours
#         - can do smoothed caiman contours or just the normal pixelwise masks?

import sys, os
import subprocess as sp
import tifffile as tfl
from glob import glob
import numpy as np
from scipy import stats
from scipy import sparse
from skimage import io, transform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from mworksbehavior import mwkfiles
from mworksbehavior.imaging import intrinsic as ii
import mworksbehavior as mwb
import mworksbehavior.mwk_io
import pytoolsMH as ptMH

# from caiman.utils.visualization import get_contours
sys.path.append(os.path.expanduser('~/Repositories/CaImAn/caiman/source_extraction/cnmf/'))
from deconvolution import *

# predefine handy operations
a_ = np.asarray
r_ = np.r_
n_ = np.newaxis


#### utility class for handling caiman output
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


#### general 2p analysis class - handles all the common quick analyses we do    
class analysis_2p_general:
    
    def __init__(self,file_path,expt_params):
        '''initialze class'''
        save_file_path = os.path.join(file_path,'analysisGeneral')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path, exist_ok=True)
        
        self.save_file_path = save_file_path
        self.expt_params = expt_params
        self.fps = expt_params['fps']
        self.nTrial = expt_params['nTrial']
        self.nFrTrial = expt_params['nFrTrial']
        self.nPreFr = expt_params['nPreFr']
        self.nStimFr = expt_params['nStimFr']
        self.nPostFr = expt_params['nPostFr']
        self.baseline_range = expt_params['baseline_range']
        self.nCell = expt_params['nCell']
        
        # two parameters relating to masks are saved in the generate_cell_masks_XXXX functions
        # self.file_path_masks - retains the path to the file used to generate the masks from
        # self.cell_masks - retains the actual mask arrays used on this data set, also returned by the above function
        
        
    def __enter__(self):
        return self

    
    def __exit__(self,exc_type,exc_value,traceback):
        return

    
    def generate_average_intensity_projection(self,im):
        '''calculate the average intensity projection of an image stack'''
        aip = im.mean(axis=0)
        save_path = os.path.join(self.save_file_path,'aip.tif')
        tfl.imsave(save_path,aip,bigtiff=True)
        print('Average intensity projection created. Saved at '+save_path)
        return aip
    
    
    def generate_cell_masks_caiman(self,file_path_masks,xDim=256,yDim=256,threshold=0.075):
        '''grabs the weighted cell component masks from caiman output and binarizes them; returns masks as shape (yDim,xDim,nCell)'''
        cnm_masks = load_caiman_output(file_path_masks)
        spatial = cnm_masks.A.toarray()
        spatial_ims = spatial.reshape((yDim,xDim,spatial.shape[1]))
        cell_masks = np.zeros((spatial_ims.shape))
        for i in range(spatial.shape[1]):
            m = spatial_ims[:,:,i]
            m[m>threshold]=1
            m[m<threshold]=0
            cell_masks[:,:,i] = m.T
        print('Number of masks: ',cell_masks.shape[2])
        assert cell_masks.shape[2] == self.nCell, 'Number of cell masks computed does not equal number specified in experiment parameters'
        self.file_path_masks = file_path_masks
        self.cell_masks = cell_masks
        return cell_masks

    
    def generate_cell_masks_suite2p(self,file_path_masks,xDim=256,yDim=256):
        '''pulls cell component masks from suite2p output; returns masks as shape (yDim,xDim,nCell)'''
        # get file_names for necessary file data
        suite2p_results_file = os.path.join(file_path_masks,'analysisSuite2p/suite2p/plane0','stat.npy')
        suite2p_iscell_file = os.path.join(file_path_masks,'analysisSuite2p/suite2p/plane0','iscell.npy')
        # read in file data
        results_dict = np.load(suite2p_results_file,allow_pickle=True)
        iscell = np.load(suite2p_iscell_file,allow_pickle=True)
        # calculate the number of accepted cells from suite2p analysis
        nCell = len(np.where(iscell[:,0]==1)[0])
        # create blank mask frames
        cell_masks = np.zeros((yDim,xDim,nCell))
        # iterate through every cell, determine if it is accepted, and create mask frame accordingly
        # accepted and rejected components are randomly assigned indices, so must use counter for indexing final array
        nAcceptedComponentsCounter = 0
        for iC in range(len(iscell)):
            if iscell[iC,0] == 1:
                yPix = results_dict[iC]['ypix']
                xPix = results_dict[iC]['xpix']
                cell_masks[yPix,xPix,nAcceptedComponentsCounter] = 1
                nAcceptedComponentsCounter += 1 # iterate counter for indexing next accepted component
        print('Number of masks: ',cell_masks.shape[2])
        assert cell_masks.shape[2] == self.nCell, 'Number of cell masks computed does not equal number specified in experiment parameters'
        self.file_path_masks = file_path_masks
        self.cell_masks = cell_masks
        return cell_masks


    def _im_mask_and_avg(self,im,masks):
        '''applies a series of masks to an image stack and returns the average value of the
        image stack within each mask across all the frames in the stack; return shape of (nCell,nFrame)'''
        nFr, nRows, nCols = im.shape
        im_vect = np.reshape(im,(nFr,nRows*nCols))
        mask_vect = np.reshape(masks,(nRows*nCols,masks.shape[2]))
        mask_sizes = mask_vect.sum(axis=(0))

        mask_vect_sparse = sparse.csr_matrix(mask_vect)
        masked_im_sums = sparse.csr_matrix.dot(im_vect,mask_vect_sparse)
        masked_im_avgs = masked_im_sums/mask_sizes[n_,:] # shape (nFrame,nCell)

        return masked_im_avgs.T # shape (nCell,nFrame)


    def generate_trialavg_dfof_frame(self,im,save_ims=True):
        '''generate the trial-averaged dF/F frames for a given image stack'''
         # set dF/F parameters
        nTrial = self.nTrial
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        br = self.baseline_range
        
        print('Generating frame-based trial-averaged dF/F...')

        # parse image in trials and frames per trial
        im_raw = im.reshape((nTrial,nFrTrial,im.shape[1],im.shape[2]))

        # background subtraction value
        im_bg = im_raw[:,:nPreFr,:,:].min()
        im_corrected = np.subtract(im_raw,im_bg)

        # calculate trial average F map
        trialAvgF = np.mean(im_corrected,axis=(0)) # returns as shape (nFrTrial,nXdim,nYdim)

        # calculate trial average dF/F map; returns as shape (nFrTrial,nXdim,nYdim)
        trialAvgBaseF = np.mean(trialAvgF[br[0]:br[1],:,:],axis=(0)) # average frame for baseline
        trialAvgDF = np.zeros(trialAvgF.shape)
        trialAvgDFoF = np.zeros(trialAvgF.shape)
        for iF in range(trialAvgF.shape[0]):
            trialAvgDF[iF,:,:] = (trialAvgF[iF,:,:]-trialAvgBaseF[n_,:,:])
            trialAvgDFoF[iF,:,:] = ((trialAvgF[iF,:,:]-trialAvgBaseF[n_,:,:])/(trialAvgBaseF[n_,:,:]))*100
            
        # save image stacks
        if save_ims:
            tfl.imsave(os.path.join(self.save_file_path,'trialAvgF.tif'),trialAvgF,bigtiff=True)
            tfl.imsave(os.path.join(self.save_file_path,'trialAvgDF.tif'),trialAvgDF,bigtiff=True)
            tfl.imsave(os.path.join(self.save_file_path,'trialAvgDFoF.tif'),trialAvgDFoF,bigtiff=True)
            tfl.imsave(os.path.join(self.save_file_path,'trialAvgBaseF.tif'),trialAvgBaseF,bigtiff=True)
            print('Resulting trial-averaged stacks saved at '+self.save_file_path)
            
        print('Frame-based trial-averaged dF/F completed.')

        return trialAvgDFoF,trialAvgDF,trialAvgF,trialAvgBaseF
    
    
    def generate_dfof_cell_traces(self,im,cell_masks):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''
        # set dF/F parameters
        nTrial = self.nTrial
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        nCell = self.nCell
        br = self.baseline_range
        
        print('Applying masks to image and generating cell traces...')
        
        # find background subtraction value from across pre-stim frames (minimum pixel value)
        im_raw = im.reshape((nTrial,nFrTrial,im.shape[1],im.shape[2]))
        im_bg = im_raw[:,:nPreFr,:,:].min()

        # use cell_masks to define raw caclium trace for a given cell region (average across masked region)
        raw_traces = self._im_mask_and_avg(im,cell_masks) # returns as shape (nCell,nFrames)
        # subtract background value from all traces
        F = raw_traces - im_bg

        # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
        baseF = F.reshape((nCell,nTrial,nFrTrial))[:,:,br[0]:br[1]].mean(axis=(1,2))

        # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
        dF = np.zeros(F.shape)
        dFoF = np.zeros(F.shape)
        for iC in range (nCell):
            for iF in range (nTrial*nFrTrial):
                dF[iC,iF] = F[iC,iF]-baseF[iC]
                dFoF[iC,iF] = (dF[iC,iF]/baseF[iC]) * 100
            
        print('Masked cell traces completed.')
                    
        return dFoF,dF,F,baseF
        
        
    def compute_denconvolved_cell_traces(self,F,method='foopsi'):
        '''compute the deconvolved traces from fluorescence traces for individual cells
        can use |foopsi| or |oasis| methods within caiman toolbox'''
        # F shape: (nCell,nTrial,nFrTrial)
        nCell = F.shape[0]
        
        # initialize variables for storring inferred denoised F and spike rate
        denoised_F = np.zeros(F.shape)
        SR = np.zeros(F.shape) # inferred spike rate
        
        print('Computing spike deconvolution using '+method+' method...')
        
        # calculate deconvolution on each cell trace
        if method=='foopsi':
            for iC in range(nCell):
                trace = F[iC,:]
                c, _, _, _, _, sp, _ = constrained_foopsi(
                                            trace, p=2, bas_nonneg=False, noise_method='mean', 
                                            fudge_factor=.97, optimize_g=5)
                denoised_F[iC,:] = c
                SR[iC,:] = sp
        elif method=='oasis':
            for iC in range(nCell):
                trace = F[iC,:]
                c, sp, _, _, _ = oasisAR2(trace, b_nonneg=False,optimize_g=5)
                denoised_F[iC,:] = c
                SR[iC,:] = sp
        else:
            raise 'Method must be |foopsi| or |oasis|.'
        
        print('Spike deconvolution completed.')
        
        return denoised_F,SR
    
    
    def compute_trialavg_2pstim_map_frame(self,im,color_lim=50,nStimFr_cut=0,nRespFr=10,save=False):
        '''Compute a single frame trial averaged dF/F map of repeated 2p stimulation (whole frame)
        Inputs:
        - im: the input image
        - color_lim: colorscale limits (+/-) in %dF/F
        - nStimFr_cut: number of frames where stimlation artifact is present to leave out of dF/F map
        - nBaseFr: number of baseline frames to average over ([nPreFrs-nBaseFrs:nPreFrs])
        - nRespFr: number of response frames to average over
        - save: save figure of map plot
        '''
        print('Computing single frame trial-averaged dF/F response map...')
        
        # set dF/F parameters
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        nTrial = self.nTrial
        br = self.baseline_range

        # calculate dF/F map
        im_raw = im.reshape(nTrial,nFrTrial,im.shape[1],im.shape[2]) # reshape image array
        im_corrected = im_raw - im_raw[:,:nPreFr,:,:].min() # background subtraction
        base = np.mean(im_corrected[:,br[0]:br[1],:,:],axis=(0,1)) # average for baseline
        resp = np.mean(im_corrected[:,nPreFr+nStimFr_cut:nPreFr+nStimFr_cut+nRespFr,:,:],axis=(0,1)) # average for response
        dFoF_map_im = ((resp-base)/(base))*100 # dF/F calculation and conversion to percent change

        # plot dfof map
        cmap = 'RdBu_r'
        v = color_lim
        
        dFoF_map_fig = plt.figure(figsize=[12,9])
        plt.imshow(dFoF_map_im,cmap=cmap,vmin=-v,vmax=v)
        
        cb = plt.colorbar(ticks=[-v,0,v])
        cb.ax.tick_params(length=0,labelsize=15)
        
        plt.xticks([i*32 for i in range(9)],[])
        plt.yticks([i*32 for i in range(9)],[])
        plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)
        
        plt.title('%% dF/F Map; %s Reps; BaseFrRange=%s; nRespFr=%s' % (nTrial,br,nRespFr),fontsize=16)

        if save:
            save_path = os.path.join(self.save_file_path,'trialavg_2pstim_map_frame.png')
            plt.savefig(save_path,bbox_inches='tight')
            print('Single frame trial-averaged dF/F response map completed. Figure saved at '+save_path)
        else:
            print('Single frame trial-averaged dF/F response map completed.')
        
        return dFoF_map_im
    
    
    def compute_trialavg_2pstim_map_cell(self,im,dFoF_traces,cell_masks,color_lim=50,nStimFr_cut=0,nRespFr=10,save=False):
        '''Compute a single trial averaged dF/F map value for each cell mask of repeated 2p stimulation
        Inputs:
        - im: the input image
        - dFoF: the computed dFoF trace of every cell
        - cell_masks: the spatial cell_masks corresponding to the dFoF traces
        - color_lim: colorscale limits (+/-) in %dF/F
        - nStimFr_cut: number of frames where stimlation artifact is present to leave out of dF/F map
        - nRespFr: number of response frames to average over for single response value in map
        - save: save figure of map plot
        '''
        print('Computing single frame trial-averaged dF/F response map...')
        
        # set dF/F parameters
        nCell = self.nCell
        nTrial = self.nTrial
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        br = self.baseline_range
        
        # reshape dFoF by trials and frames in trial
        dFoF = dFoF_traces.reshape(nCell,nTrial,nFrTrial)
        
        # calculate trial average dF/F response for each cell
        trialavg_dFoF_trace = dFoF.mean(axis=1) # trial averaging; shape (nCell,nFrTrial) 
        trialavg_dFoF_val = np.mean(trialavg_dFoF_trace[:,nPreFr+nStimFr_cut:nPreFr+nStimFr_cut+nRespFr],axis=(1)) # dF/F response value
        
        # generate map of avg dF/F for each cell mask
        nRows = im.shape[1]
        nCols = im.shape[2]
        nCell = self.nCell
        dFoF_map_cell_masks = np.zeros((nCell,nRows,nCols))
        for iC in range(nCell):
            cell_map = trialavg_dFoF_val[iC] * cell_masks[:,:,iC]
            cell_map[cell_map==0] = np.nan
            dFoF_map_cell_masks[iC,:,:] = cell_map
            
        # plot cell mask dF/F map
        cmap = 'RdBu_r'
        v = color_lim
        
        plt.figure(figsize=[12,9])
        for iC in range(nCell):
            plt.imshow(dFoF_map_cell_masks[iC,:,:],cmap=cmap,vmin=-v,vmax=v)

        cb = plt.colorbar(ticks=[-v,0,v])
        cb.ax.tick_params(length=0,labelsize=15)
        
        plt.xticks([i*32 for i in range(9)],[])
        plt.yticks([i*32 for i in range(9)],[])
        plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)
        
        plt.title('%% dF/F Map; %s Reps; BaseFrRange=%s; nRespFr=%s' % (nTrial,br,nRespFr),fontsize=16)

        if save:
            save_path = os.path.join(self.save_file_path,'trialavg_2pstim_map_cellmask.png')
            plt.savefig(save_path,bbox_inches='tight')
            print('Trial-averaged dF/F response map for cell masks completed. Figure saved at '+save_path)
        else:
            print('Trial-averaged dF/F response map for cell masks completed.')  
            
        return dFoF_map_cell_masks,trialavg_dFoF_val



# ### utility functions

# # fxns for dealing with caiman, suite2p, or mworks data

def load_caiman_output(file_path):
    '''load the caiman results and return as an object'''
    caiman_results_v3 = os.path.join(file_path,'analysisCaiman/','results-analysis-v3.npz')
    cnm_dict = np.load(caiman_results_v3,allow_pickle=True)['results_dict'].item()
    return objectview(cnm_dict)


def compute_caiman_raw_traces(cnm):
    '''compute raw data trace for each cell component from caiman output'''
    b_weighted_by_cell = np.dot(cnm.b.T,cnm.A.toarray())
    background_cell = np.dot(cnm.f.T,b_weighted_by_cell)
    raw_traces = background_cell.T + cnm.YrA + cnm.C
    return cnm_raw_traces


def load_suite2p_output(file_path):
    return True


def load_mwk_output(file_path,stim_range='all'):
    '''load the .mwk file to extract mworks experiment parameters'''
    try:
        mwkfile = glob(file_path+'/*.mwk*')[0]
    except:
        raise OSError('\nNo .mwk file found at specified path. Make sure path is defined correctly.')
    mwkname = os.path.join(file_path,mwkfile)
    # for fixing up corrupt files
    useStimRange = stim_range
    # get imaging constants
    dd2 = mwb.imaging.consts.DataDir2p(file_path)
    print(dd2)
    # generate the h5 file from mwk
    mworksbehavior.mwk_io.mwk_to_h5(
        mwkname,
        keep_system_vars=False, 
        exist_delete=True,
    )
    # read mworks file
    try:
        mwf = mwkfiles.RetinotopyMap2StimMWKFile(dd2.h5name,stims_to_keep=useStimRange)
    except:
        mwf = mwkfiles.RetinotopyMap1MWKFile(dd2.h5name,stims_to_keep=useStimRange)
    mwf.save_stim_params(dd2.h5stimsname)
    mwf.compute_imaging_constants()
    os.rmdir(os.path.join(file_path,'analysisStacks')) # should really change underlying code to not create this dir..
    print(mwf.constS)
    print('nStims: %d, nFramesPerStim: %d, nReps: %d' % (mwf.nstim, mwf.nframes_stim, mwf.nreps))
    return mwf,dd2


def parse_mwk_levels(mwf):
    '''generate a list containing lists of frames for each stim level for an MWorks experiment'''
    levels = np.unique(mwf.df[mwf.levelVar])
    levelFr = np.zeros((len(levels),int(mwf.nframes_stim*mwf.nreps)))
    for (iL,tL) in enumerate(levels):
        desIdx = np.where(mwf.stimDf[mwf.levelVar].values==tL)[0]
        levelFr_idx = []
        for idx in desIdx:
            levelFr_idx+=(list(range(mwf.nframes_stim*idx,mwf.nframes_stim*idx+(mwf.nframes_stim))))
        levelFr[iL,:] = a_(levelFr_idx)
    levelFr = levelFr.astype(int)
    nLevel = len(levels)
    return levels,levelFr,nLevel


def reshape_celltrace_by_mwk_levels(traces,expt_params,mwf):
    '''resize an array of cell traces of shape (nCell,nFrame) 
    into shape (nCell,nLevel,nTrial,nFrTrial)'''
    # parse mwk level information
    levels,levelFr,nLevel = parse_mwk_levels(mwf)
    
    # define experiment parameters
    nCell = expt_params['nCell']
    nTrial = expt_params['nTrial']
    nFrTrial = expt_params['nFrTrial']

    # intialize cell traces reshaped by levels and trials
    traces_parsed = np.zeros((nCell, nLevel, nTrial, nFrTrial))
    
    # extract values from frames of specified level and trial
    for iC in range(nCell):
        for iL in range(nLevel):
            for iT in range(nTrial):
                start = (iT*nFrTrial)
                end   = (iT*nFrTrial)+nFrTrial
                traces_parsed[iC,iL,iT,:] = var[iC,levelFr[iL][start:end]]
    
    return traces_parsed


def reshape_im_by_mwk_levels(im,expt_params,mwf):
    '''resize an image stack of shape (nFrame,nXdim,nYdim) 
    into shape (nLevel,nTrial,nFrTrial,nXdim,nYdim)'''
    # parse mwk level information
    levels,levelFr,nLevel = parse_mwk_levels(mwf)
    
    # define experiment parameters
    nTrial = expt_params['nTrial']
    nFrTrial = expt_params['nFrTrial']
    
    # intialize cell traces reshaped by levels and trials
    im_parsed = np.zeros((nLevel, nTrial, nFrTrial, im.shape[1], im.shape[2]))
    
    # extract values from frames of specified level and trial
    for iL in range(nLevel):
        for iT in range(nTrial):
            start = (iT*nFrTrial)
            end   = (iT*nFrTrial)+nFrTrial
            im_parsed[iL,iT,:,:,:] = im[levelFr[iL][start:end],:,:]
    
    return im_parsed
