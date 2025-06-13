# ##### TO-DO:

# - implement handling of data with multiple levels (i.e. mworks trial types)
# - within analysis_2p_general class
#     - save image of aip with masks superimposed? (indicate stim vs. no-stim and number)
# - plotting fxns section???
#     - make a function to plot cell contours
#         - can do smoothed caiman contours or just the normal pixelwise masks?

import sys, os
import subprocess as sp
from datetime import datetime

import tifffile as tfl
from glob import glob
import math as math
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from scipy.ndimage import gaussian_filter as gaussian_filter
from skimage import io, transform

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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
        self.zoom = expt_params['zoom']
        self.frameSz = expt_params['frameSz']
        self.nTrial2P = expt_params['nTrial2P']
        self.nTrialV = expt_params['nTrialV']
        self.nFrTrial = expt_params['nFrTrial']
        self.nPreFr = expt_params['nPreFr']
        self.nStimFr = expt_params['nStimFr']
        self.nPostFr = expt_params['nPostFr']
        self.baseline_range = expt_params['baseline_frame_range']
        self.nHoloFr = expt_params['nHoloFr']
        self.nCell = expt_params['nCell']
        self.holoPatterns = expt_params['holoPatterns']

        
        # two parameters relating to masks are saved in the generate_cell_masks_XXXX functions
        # self.file_path_masks - retains the path to the file used to generate the masks from
        # self.cell_masks - retains the actual mask arrays used on this data set, also returned by the above function  
        
    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        return

    def generate_aip(self,im):
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
        suite2p_results_file = os.path.join(file_path_masks,'suite2p/plane0','stat.npy')
        suite2p_iscell_file = os.path.join(file_path_masks,'suite2p/plane0','iscell.npy')
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
    
    def calculate_cell_pattern_coords(self,file_path_pattern,cell_masks,csv_pattern_filename='1.csv',downscale_factor=2):
        '''calculate coordinates of cell mask center of masses and coordinates of stim pattern targets'''
        # get coordinates of center of masses of masks in pixels
        print('zoom =',self.zoom, end='. ')
        nCell = self.nCell
        cell_coords = []
        for iC in range(nCell):
            ys,xs = np.where(cell_masks[:,:,iC]>0)
            coord = [np.average(xs),np.average(ys)]
            cell_coords.append(coord)
        cell_coords = np.asarray(cell_coords)
        # get coordinates of pattern targets in microns
        pattern_file = os.path.join(file_path_pattern,csv_pattern_filename) # TODO: handle multiple files at once?
        df = pd.read_csv(pattern_file)
        coords = np.rollaxis(np.asarray((df['X'],df['Y'])),1,0)
        # convert from microns to pixels (hard-coded numbers based on SI defaults)
        # and scale to match downscaled image
        conversion_factor = ((512/downscale_factor)/(1037/self.zoom)) # units of pixel/micron #you don't need to do the pixel to micron conversion bc you use pngs not tiffs when drawing on yantings gui
        pix_coords = coords*conversion_factor
        # remove first entry - zero order position
        pattern_coords = pix_coords[1:]
        return cell_coords,pattern_coords,conversion_factor
    
    def calc_pattern_coords_DLpts(self,file_path_pattern,cell_masks,whichPtsInPat,csv_pattern_filename='1.csv',downscale_factor=2):
        '''calculate coordinates of cell mask center of masses and coordinates of stim pattern targets for diffraction limited point patterns'''
        # get coordinates of center of masses of masks in pixels
        print('zoom =',self.zoom, end='. ')
        nCell = self.nCell
        cell_coords = []
        for iC in range(nCell):
            ys,xs = np.where(cell_masks[:,:,iC]>0)
            coord = [np.average(xs),np.average(ys)]
            cell_coords.append(coord)
        cell_coords = np.asarray(cell_coords)
        # get coordinates of pattern targets in microns
        pattern_file = os.path.join(file_path_pattern,csv_pattern_filename) # TODO: handle multiple files at once?
        
        
        df = pd.read_csv(pattern_file)
        coords = np.rollaxis(np.asarray((df['X'],df['Y'])),1,0)
        
        coords_pat0 = coords[whichPtsInPat[0,0]:whichPtsInPat[0,1]]
        coords_pat1 = coords[whichPtsInPat[1,0]:whichPtsInPat[1,1]]
        coords_pat2 = coords[whichPtsInPat[2,0]:whichPtsInPat[2,1]]
        coords_pat3 = coords[whichPtsInPat[3,0]:whichPtsInPat[3,1]]
        coords_pat4 = coords[whichPtsInPat[4,0]:whichPtsInPat[4,1]]

        # convert from microns to pixels (hard-coded numbers based on SI defaults)
        # and scale to match downscaled image
        conversion_factor = ((512/downscale_factor)/(1037/self.zoom)) # units of pixel/micron #you don't need to do the pixel to micron conversion bc you use pngs not tiffs when drawing on yantings gui
        
        pattern_coords = coords*conversion_factor         # remove first entry - zero order position for non dlp pts
        
        return cell_coords,pattern_coords,conversion_factor
    
       
    def label_stimulated_cells(self,cell_coords,pattern_coords,radius_microns=10,downscale_factor=2):
        '''compare coords of pattern targets and cells to find which cells are directly stimulated 
        within a radius of the pattern target coordinates'''
        print('Finding cell IDs within direct stimulation radius...', end=' ')
        # convert radius_microns to radius_pix
        conversion_factor = ((512/downscale_factor)/(1037/self.zoom)) # units of pixel/micron
        radius_pix = radius_microns*conversion_factor
        # find within-radius cell_coords
        nCell = self.nCell
        stim_iC = []
        for iP in range(pattern_coords.shape[0]):
            pattern_coord = pattern_coords[iP,:]
            euclidean_distances = np.linalg.norm(cell_coords-pattern_coord,axis=1)
            within_radius_iC = np.where(euclidean_distances<radius_pix)[0]
            [stim_iC.append(iC) for iC in within_radius_iC]
        # get unqiue values from list as stimulated cell indices  
        stim_iC = np.unique(np.asarray(stim_iC))
        print('Stimulated cell IDs found.')
        # annotate the cells which are stimulated vs. non-stimulated
        all_iC = list(range(nCell))
        nostim_iC = [iC for iC in all_iC if iC not in stim_iC]
        return stim_iC,nostim_iC,all_iC
    
    def label_stimulated_cells_DLP(self,cell_coords,pattern_coords,whichPtsInPat,iPat=0,radius_microns=10,downscale_factor=2):
        '''compare coords of pattern targets and cells to find which cells are directly stimulated 
        within a radius of the pattern target coordinates'''
        print('Finding cell IDs within direct stimulation radius...', end=' ')
        # convert radius_microns to radius_pix
        conversion_factor = ((512/downscale_factor)/(1037/self.zoom)) # units of pixel/micron
        radius_pix = radius_microns*conversion_factor
        # find within-radius cell_coords
        nCell = self.nCell
        stim_iC = []
        pattern_coords_stimd = pattern_coords[whichPtsInPat[iPat,0]:whichPtsInPat[iPat,1], :]
        print(whichPtsInPat[iPat,0])
        print(whichPtsInPat[iPat,1])
        for iP in range(pattern_coords_stimd.shape[0]):
            pattern_coord = pattern_coords_stimd[iP,:]
            euclidean_distances = np.linalg.norm(cell_coords-pattern_coord,axis=1)
            within_radius_iC = np.where(euclidean_distances<radius_pix)[0]
            [stim_iC.append(iC) for iC in within_radius_iC]
        # get unqiue values from list as stimulated cell indices  
        stim_iC = np.unique(np.asarray(stim_iC))
        print('Stimulated cell IDs found.')
        # annotate the cells which are stimulated vs. non-stimulated
        all_iC = list(range(nCell))
        nostim_iC = [iC for iC in all_iC if iC not in stim_iC]
        return stim_iC,nostim_iC,all_iC

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
    
    def plot_tuningcurve_cell (self,cell_iD,dfoF_trace_lowessV,RespFrames=10):

        nPreFrV = self.nPreFrV
        nStimFrV = self.nStimFrV

        #get single df/f values for each cell using an average over end of stim frames and lowessed cell mask trace
        dfoF_V_avg = dfoF_trace_lowessV[:, :, nPreFrV+nStimFrV-RespFrames:nPreFrV+nStimFrV].mean(axis=2)

        xs = [0,np.pi*0.25,np.pi*0.5,np.pi*0.75,np.pi,np.pi*1.25,np.pi*1.5,np.pi*1.75, np.pi*2]
        dfoF_Vplot = dfoF_V_avg[:, cell_iD]
        dfoF_Vplot = np.concatenate((dfoF_Vplot, dfoF_Vplot[:1, :]))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(xs,dfoF_Vplot,color='blue')
        ax.set_rmax(30)
        ax.set_rticks([10, 20, 30])  # Less radial ticks
        ax.set_xlabel('df/f')
        ax.set_rlabel_position(-20)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.set_title('cell '+str(cell_iD[0]), va='bottom')
        plt.show()
        
        return

    def generate_trialavg_dfof_frame(self,im,save_ims=True):
        '''generate the trial-averaged dF/F frames for a given image stack'''
         # set dF/F parameters
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        br = self.baseline_range
        
        print('Generating frame-based trial-averaged dF/F...')

        # parse image in trials and frames per trial
        im_raw = im.reshape((nTrial2P,nFrTrial,im.shape[1],im.shape[2]))

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
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        nCell = self.nCell
        br = self.baseline_range
        
        print("Began applying masks to image and generating cell traces at", datetime.now().time(), "...")
        
        # find background subtraction value from across pre-stim frames (minimum pixel value)
        im_raw = im.reshape((nTrial2P,nFrTrial,im.shape[1],im.shape[2]))
        #im_bg = im_raw[:,:nPreFr,:,:].min() #w/ downtick & blanking need the min of all frames
        print('Calculating min...')
        im_bg = np.min(np.mean(im_raw, axis=(0,1)))
        print('Minimum is ', im_bg)
        
        # use cell_masks to define raw calcium trace for a given cell region (average across masked region)
        raw_traces = self._im_mask_and_avg(im,cell_masks) # returns as shape (nCell,nFrames)
        if im_bg > 3:
            # subtract background value from all traces
            print("Background noise; subtracting min values from traces...", datetime.now().time())
            F_cell = raw_traces - im_bg
        else:
            F_cell = raw_traces

        # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
        print('Calculating baseline values...', datetime.now().time())
        baseF = F_cell.reshape((nCell,nTrial2P,nFrTrial))[:,:,br[0]:br[1]].mean(axis=(1,2))

        # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
        dF_cell = np.zeros(F_cell.shape)
        dFoF_cell = np.zeros(F_cell.shape)
        for iC in range (nCell):
            if iC % 50 == 0:
                print(iC, end=' ')
            for iF in range (nTrial2P*nFrTrial):
                dF_cell[iC,iF] = F_cell[iC,iF]-baseF[iC]
                dFoF_cell[iC,iF] = (dF_cell[iC,iF]/baseF[iC]) * 100
            
        print('Masked cell traces completed.')
                    
        return im_raw,im_bg,dFoF_cell,dF_cell,F_cell,baseF
    
    def generate_dfof_cell_traces_indivBL(self,im,cell_masks,stimOrder,F_s2p=np.array([]),use_suite2P_trace=False,nStimFr_cut=0,nRespFr=10):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''

        # set dF/F parameters
        holoPats = self.holoPatterns #still works if only 1 pattern, stim
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        nCell = self.nCell
        br = self.baseline_range
    
        # find background subtraction value from across pre-stim frames (minimum pixel value)
        aip = np.mean(im, axis=0)
        im_bg = np.min(aip) # shape is orientation,trials,trialframes,sidelength,sidelength
        print('95th percentile = ', np.percentile(aip, 95)) # shape is orientation,trials,trialframes,sidelength,sidelength
        print('minimum = ', im_bg)
        # use cell_masks to define raw calcium trace for a given cell region (average across masked region)
        if use_suite2P_trace == False:
            raw_traces = self._im_mask_and_avg(im,cell_masks) # returns as shape (nCell,nFrames)
        else:
            raw_traces = F_s2p
        print(raw_traces.shape)
        
        if im_bg > 1:
            # subtract background value from all traces
            print("Subtracting min values from traces...", datetime.now().time())
            F = raw_traces - im_bg
        else:
            F = raw_traces
        F_reshape = F.reshape(nCell, nTrial2P, nFrTrial)
    
        avgTrial = np.zeros(shape = (holoPats, nCell, nFrTrial))
        trial_dfoF_trace2P = np.zeros(avgTrial.shape)
        df_vis_cell = np.zeros(shape = (holoPats, nCell))
        dfoF_vis_cell = np.zeros(df_vis_cell.shape)
        F_holo = np.zeros(shape = (holoPats, nCell, nTrial2P, nFrTrial))

        # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
        for stims in range(len(stimOrder)):
            trialForRandom = math.floor(stims/holoPats) #which iteration of the pattern are you on? If 80 trials per pattern, this will go to 80
            F_holo[int(stimOrder[stims]-1), :, trialForRandom, :] = F_reshape[:, stims, :] # stim, cell, trial, frame            
            
        avgTrial_F = np.mean(F_holo, axis=2) #returns shape holos, nCells, trialframes
        avgPrestim = np.mean(avgTrial_F[:, :, br[0]:br[1]], axis=2) #returns shape nLevels,nCells
        avgStim = np.mean(avgTrial_F[:, :, br[1]+nStimFr_cut:br[1]+nStimFr_cut+nRespFr], axis=2) #returns shape nLevels, nCells

        df_vis_cell = avgStim - avgPrestim
        dfoF_vis_cell = df_vis_cell / (avgPrestim)*100
        
        print(avgPrestim.shape)
        
        avgPrestim_bc = avgPrestim.reshape(avgPrestim.shape[0], avgPrestim.shape[1], 1) #broadcasting so it is same ndims as avgTrial_F
        trial_dfoF_trace2P =  (avgTrial_F - avgPrestim_bc) / avgPrestim_bc*100
        # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
        #dF = np.zeros(F.shape)
        #dFoF = np.zeros(F.shape)
        #for iC in range (nCell):
        #    for iF in range (nTrial2P*nFrTrial):
        #        dF[iC,iF] = F[iC,iF]-baseF[iC]
        #        dFoF[iC,iF] = (dF[iC,iF]/baseF[iC]) * 100

        return avgTrial_F, im_bg, avgPrestim, trial_dfoF_trace2P, df_vis_cell, dfoF_vis_cell, F_holo

    
    
    def _compute_spike_rate_norm(self,spk):
        '''calculate putative normalized spike rate using Poisson assumption (meanFR = varFR)'''
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        br = self.baseline_range
        fps = self.fps
        nCell = spk.shape[0]
        
        # get spike counts over baseline time points
        base_spk = spk.reshape((nCell,nTrial2P,nFrTrial))[:,:,br[0]:br[1]]
        base_spk = base_spk.reshape((nCell,nTrial2P*base_spk.shape[2]))
        nSamples = base_spk.shape[1]
        
        # define normalization factor and set minimum threshold
        normFactor = base_spk.var(axis=1)/base_spk.mean(axis=1)
        minThresh = np.min((np.percentile(normFactor[~np.isnan(normFactor)],10), nSamples/10))  # thresh is 10th percentile or only resp in 10 bins, whichever is smaller
        normFactor[normFactor<minThresh] = minThresh
        
        # divide spk by norm factor to return putative spike counts
        spk_norm = spk/normFactor[:,n_]
        # multiply the frames per second by spike count to return spike rate
        SR = spk_norm*fps
        
        print("Finished at", datetime.now().time())
        
        return SR
        
        
    def compute_denconvolved_cell_traces(self,F,method='foopsi'):
        '''compute the deconvolved traces from fluorescence traces for individual cells
        can use |foopsi| or |oasis| methods within caiman toolbox'''
        # F shape: (nCell,nTrial,nFrTrial)
        nCell = F.shape[0]
        
        # initialize variables for storring inferred denoised F and spike count
        denoised_F = np.zeros(F.shape)
        spk = np.zeros(F.shape) # inferred spike count
        
        print('Computing spike deconvolution using '+method+' method...')
        
        # calculate deconvolution on each cell trace
        if method=='foopsi':
            for iC in range(nCell):
                trace = F[iC,:]
                c, _, _, _, _, sp, _ = constrained_foopsi(
                                            trace, p=2, bas_nonneg=False, noise_method='mean', 
                                            fudge_factor=.97, optimize_g=5)
                denoised_F[iC,:] = c
                spk[iC,:] = sp
        elif method=='oasis':
            for iC in range(nCell):
                trace = F[iC,:]
                c, sp, _, _, _ = oasisAR2(trace, b_nonneg=False,optimize_g=5)
                denoised_F[iC,:] = c
                spk[iC,:] = sp
        else:
            raise 'Method must be |foopsi| or |oasis|.'
        
        print('Spike deconvolution completed.')
        
        SR = self._compute_spike_rate_norm(spk)
        
        print('Normalized spike rate computed.')
        
        return denoised_F,spk,SR
     
    def compute_trialavg_2pstim_map_frame(self,im,stimOrder,color_lim=50,nStimFr_cut=0,nRespFr=10,save=False):
        '''Compute a single frame trial averaged dF/F map of repeated 2p stimulation (whole frame)
        Inputs:
        - im: the input image
        - stimOrder: Takes in an array of the stimuli presented. If one stim pattern, array of 1s, length = nTrials
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
        nTrial2P = self.nTrial2P
        br = self.baseline_range
        holoPats = self.holoPatterns #still works if only 1 pattern, stim

        # calculate dF/F map
        im_raw = im.reshape(nTrial2P,nFrTrial,im.shape[1],im.shape[2]) # reshape image array
        im_corrected = im_raw - im_raw.min() # background subtraction
        im_corrected_pat = np.zeros((holoPats, int(nTrial2P/holoPats), nFrTrial, im.shape[1], im.shape[2])) # create array with 1st dim for pattern number 
        for stims in range(0, len(stimOrder)):
            trialForRandom = math.floor(stims/holoPats) #which iteration of the pattern are you on? If 80 trials per pattern, this will go to 80
            if stims % 2 == 0:
                print(stims, end=' ')
            im_corrected_pat[stimOrder[stims]-1, trialForRandom, :, :, :] = im_corrected[stims, :, :, :] # stim, trial, frame, pixel, pixel
        print('\n')
        
        base = np.mean(im_corrected_pat[:, :, br[0]:br[1],:,:],axis=(1,2)) # average for baseline
        resp = np.mean(im_corrected_pat[:, :, nPreFr+nStimFr_cut:nPreFr+nStimFr_cut+nRespFr,:,:],axis=(1,2)) # average for response
        dFoF_map_im = ((resp-base)/(base))*100 # dF/F calculation and conversion to percent change

        # plot dfof map
        cmap = 'RdBu_r'
        v = color_lim
        
        dFoF_map_fig = plt.figure(figsize=[12,9])
        for holopat in range(holoPats):
            plt.figure()
            plt.imshow(dFoF_map_im[holopat, :, :],cmap=cmap,vmin=-v,vmax=v)

            cb = plt.colorbar(ticks=[-v,0,v])
            cb.ax.tick_params(length=0,labelsize=15)

            plt.xticks([i*32 for i in range(9)],[])
            plt.yticks([i*32 for i in range(9)],[])
            plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
            plt.gca().tick_params(axis=u'both', which=u'both',length=0)

            plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial2P/holoPats)), fontsize = 16)
            plt.title('BaseFrRange=%s; nRespFr=%s; pattern ' % (br,nRespFr) + str(holopat+1), fontsize=12)

            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_2pstim_frame_pat'+str(holopat+1)+'.png')
                plt.savefig(save_path,bbox_inches='tight',dpi=400)
                print('Single frame trial-averaged dF/F response map completed. Figure saved at '+save_path)
            else:
                print('Single frame trial-averaged dF/F response map completed.')

        return dFoF_map_im
    
    def plot_subtract_map_cell(self,dFoF_map_cell_masks_all,A,B,C,AandB,AandC,color_lim=50,save=False):
        '''Compute a single frame trial averaged dF/F map of repeated 2p stimulation (whole frame)
        Inputs:
        '''
        print('Subtracted response map...')
        
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        nTrial2P = self.nTrial2P
        br = self.baseline_range
        holoPats = self.holoPatterns #still works if only 1 pattern, stim
        nCell = self.nCell
        cmap = 'RdBu_r'
        v = color_lim

        co_map = dFoF_map_cell_masks_all[AandB, :, :, :] - (dFoF_map_cell_masks_all[A, :, :, :]+dFoF_map_cell_masks_all[B, :, :, :])
        ortho_map = dFoF_map_cell_masks_all[AandC, :, :, :] - (dFoF_map_cell_masks_all[A, :, :, :]+dFoF_map_cell_masks_all[C, :, :, :])
        
        plt.figure(figsize=[12,9])
        for iC in range(nCell):
            plt.imshow(ortho_map[iC,:,:],cmap=cmap,vmin=-v,vmax=v)

        cb = plt.colorbar(ticks=[-v,0,v])
        cb.ax.tick_params(length=0,labelsize=15)
        plt.xticks([i*32 for i in range(9)],[])
        plt.yticks([i*32 for i in range(9)],[])
        plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)

        #plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial/holoPats)), fontsize = 16)
        plt.title('%% dF/F; %s Reps; BaseFrs=%s; RespFr=%s; pattern ' % ((int(nTrial2P/holoPats)),br,nRespFr) + str(holopat+1), fontsize=16)

        if save:
            save_path = os.path.join(self.save_file_path,'nonlin_ortho'+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Subtract map completed. Figure saved at '+save_path)
        else:
            print('Subtract dF/F response map completed.')
        #################################################
        plt.figure(figsize=[12,9])
        for iC in range(nCell):
            plt.imshow(co_map[iC,:,:],cmap=cmap,vmin=-v,vmax=v)

        cb = plt.colorbar(ticks=[-v,0,v])
        cb.ax.tick_params(length=0,labelsize=15)
        plt.xticks([i*32 for i in range(9)],[])
        plt.yticks([i*32 for i in range(9)],[])
        plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
        plt.gca().tick_params(axis=u'both', which=u'both',length=0)

        #plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial/holoPats)), fontsize = 16)
        plt.title('%% dF/F; %s Reps; BaseFrs=%s; RespFr=%s; pattern ' % ((int(nTrial2P/holoPats)),br,nRespFr) + str(holopat+1), fontsize=16)

        plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial2P/holoPats)), fontsize = 16)
        plt.title('BaseFrRange=%s; nRespFr=%s; pattern ' % (br,nRespFr) + str(holopat+1), fontsize=12)

        if save:
            save_path = os.path.join(self.save_file_path,'nonlin_co'+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Subtract map completed. Figure saved at '+save_path)
        else:
            print('Subtract dF/F response map completed.')

        return

     
        
    def compute_trialavg_2pstim_map_cell(self,im,stimOrder,dFoF_traces,cell_masks,color_lim=20,nStimFr_cut=0,nRespFr=10,save=False):
        '''Compute a single trial averaged dF/F map value for each cell mask of repeated 2p stimulation
        Inputs:
        - im: the input image
        - stimOrder: Takes in an array of the stimuli presented. If one stim pattern, array of 1s, length = nTrials
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
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nPreFr = self.nPreFr
        br = self.baseline_range
        holoPats = self.holoPatterns #still works if only 1 pattern, stim
        
        # reshape dFoF by trials and frames in trial
        dFoF = dFoF_traces.reshape(nCell,nTrial2P,nFrTrial)
        dFoF_pat = np.zeros((holoPats, nCell, int(nTrial2P/holoPats), nFrTrial))
        
        for stims in range(0, len(stimOrder)):
            trialForRandom = math.floor(stims/holoPats) #which iteration of the pattern are you on? If 80 trials per pattern, this will go to 80
            dFoF_pat[int(stimOrder[stims]-1), :, trialForRandom, :] = dFoF[:, stims, :] # stim, cell, trial, frame
        
        # calculate trial average dF/F response for each cell
        trialavg_dFoF_trace2P = dFoF_pat.mean(axis=2) # trial averaging; shape (holoPats,nCell,nFrTrial) 
        trialVar_dFoF_trace2P = dFoF_pat.var(axis=2)
        trialavg_dFoF_val = np.mean(trialavg_dFoF_trace2P[:,:,nPreFr+nStimFr_cut:nPreFr+nStimFr_cut+nRespFr],axis=(2)) # dF/F response value

        # plot cell mask dF/F map
        cmap = 'RdBu_r'
        v = color_lim
        nRows = im.shape[1]
        nCols = im.shape[2]
        nCell = self.nCell
        dFoF_map_cell_masks_all = np.zeros((holoPats, nCell,nRows,nCols))
        
        for holopat in range(holoPats):
            
            # generate map of avg dF/F for each cell mask
            dFoF_map_cell_masks = np.zeros((nCell,nRows,nCols))
            for iC in range(nCell):
                cell_map = trialavg_dFoF_val[holopat, iC] * cell_masks[:,:,iC]
                cell_map[cell_map==0] = np.nan
                dFoF_map_cell_masks[iC,:,:] = cell_map
            
            dFoF_map_cell_masks_all[holopat,:,:,:] = dFoF_map_cell_masks
            plt.figure(figsize=[12,9])
            for iC in range(nCell):
                plt.imshow(dFoF_map_cell_masks[iC,:,:],cmap=cmap,vmin=-v,vmax=v)

            cb = plt.colorbar(ticks=[-v,0,v])
            cb.ax.tick_params(length=0,labelsize=15)
            plt.xticks([i*32 for i in range(9)],[])
            plt.yticks([i*32 for i in range(9)],[])
            plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
            plt.gca().tick_params(axis=u'both', which=u'both',length=0)

            #plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial/holoPats)), fontsize = 16)
            plt.title('%% dF/F; %s Reps; BaseFrs=%s; RespFr=%s; pattern ' % ((int(nTrial2P/holoPats)),br,nRespFr) + str(holopat+1), fontsize=16)
            
            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_2pstimCell_framePat'+str(holopat+1)+'.png')
                plt.savefig(save_path,bbox_inches='tight',dpi=400)
                print('Trial-averaged dF/F response map for cell masks completed. Figure saved at '+save_path)
            else:
                print('Trial-averaged dF/F response map for cell masks completed.')

        return dFoF_pat,dFoF_map_cell_masks_all,trialavg_dFoF_trace2P,trialVar_dFoF_trace2P,trialavg_dFoF_val
    
    def generate_trialavg_deconv(self,stimOrder,spikeEst_holo):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''

        # set dF/F parameters
        holoPats = self.holoPatterns #still works if only 1 pattern, stim
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        nCell = self.nCell

        avgTrial_dc = np.zeros(shape = (holoPats, nCell, nFrTrial))
        spikeHolo_reshape = spikeEst_holo.reshape(nCell, nTrial2P, nFrTrial)
        dc_holo= np.zeros(shape = (holoPats, nCell, nTrial2P, nFrTrial))


        # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
        for stims in range(len(stimOrder)):
            trialForRandom = math.floor(stims/holoPats) #which iteration of the pattern are you on? If 80 trls per pat, this will go to 80
            dc_holo[int(stimOrder[stims]-1), :, trialForRandom, :] = spikeHolo_reshape[:, stims, :] # cell,trl,fr->stim,cell,trl,fr           
        avgTrial_dc= np.mean(dc_holo, axis=2) #returns shape holos, nCells, trialframes

        return avgTrial_dc, dc_holo

    
    def plot_2pstim_map_cell(self,im,cell_masks,dfoF_vis_cell,color_lim=20,save=False):
        
        # plot cell mask dF/F map
        cmap = 'RdBu_r'
        v = color_lim
        nRows = im.shape[1]
        nCols = im.shape[2]
        nCell = self.nCell
        dFoF_map_cell_masks_all = np.zeros((holoPats,nCell,nRows,nCols))
        
        for holopat in range(holoPats):
            
            # generate map of avg dF/F for each cell mask
            dFoF_map_cell_masks = np.zeros((nCell,nRows,nCols))
            for iC in range(nCell):
                cell_map = trialavg_dFoF_val[holopat, iC] * cell_masks[:,:,iC]
                cell_map[cell_map==0] = np.nan
                dFoF_map_cell_masks[iC,:,:] = cell_map
            
            dFoF_map_cell_masks_all[holopat,:,:,:] = dFoF_map_cell_masks
            plt.figure(figsize=[12,9])
            for iC in range(nCell):
                plt.imshow(dFoF_map_cell_masks[iC,:,:],cmap=cmap,vmin=-v,vmax=v)

            cb = plt.colorbar(ticks=[-v,0,v])
            cb.ax.tick_params(length=0,labelsize=15)
            plt.xticks([i*32 for i in range(9)],[])
            plt.yticks([i*32 for i in range(9)],[])
            plt.grid(color='lightgrey',linestyle='-',linewidth=2,alpha=0.3)
            plt.gca().tick_params(axis=u'both', which=u'both',length=0)

            #plt.suptitle('%% dF/F Map; %s Reps' % (int(nTrial/holoPats)), fontsize = 16)
            plt.title('%% dF/F; %s Reps; BaseFrs=%s; RespFr=%s; pattern ' % ((int(nTrial2P/holoPats)),br,nRespFr) + str(holopat+1), fontsize=16)
            
            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_2pstimCell_framePat'+str(holopat+1)+'.png')
                plt.savefig(save_path,bbox_inches='tight',dpi=400)
                print('Trial-averaged dF/F response map for cell masks completed. Figure saved at '+save_path)
            else:
                print('Trial-averaged dF/F response map for cell masks completed.')
        return
    
    def generate_trialavg_dfof_frame_ngf(self,im,stimOrder,RespFr=5):
        
        '''generate the trial-averaged dF/F frames for a given image stack
        titles: array of strings, length = patterns, will be assigned a plot'''
        
        print('Generating frame-based trial-averaged dF/F...')

        frameSz = self.frameSz
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        br = self.baseline_range
        nHoloFr = self.nHoloFr
        nCell = self.nCell
        holoPatterns = self.holoPatterns
        
        df_2P = np.zeros((frameSz,frameSz,holoPatterns))
        dfoF_2P = np.full(df_2P.shape, np.nan)
        dfoFsmooth_2P = np.full(df_2P.shape, np.nan)
        dfoFsmoothtrial_2P = np.full((nFrTrial,frameSz,frameSz,holoPatterns), np.nan)

        if holoPatterns > 1: #if there are multiple stim patterns, seperate into stim types
            im = np.reshape(im, (nFrTrial, nTrial2P, frameSz, frameSz), order='F')
            im_pat = np.zeros((im.shape[0], int(im.shape[1]/holoPatterns), im.shape[2], im.shape[3], holoPatterns)) #trialFrames, trials_per_pattern, framesz, framesz, patterns 
            
            for stims in range(0, len(stimOrder)):
                #which iteration of the pattern are you on? If 80 trials per pattern, this will go to 80
                trialForRandom = math.floor(stims/holoPatterns)
                if stims % 50 == 0:
                    print('\nTrial number... ' ,stims, ' ',datetime.now().time(), end=' ')
                im_pat[:, trialForRandom, :, :, stimOrder[stims]-1] = im[:, stims, :, :] #frame, trial, pixel, pixel
            print("\nFinished im_pat ", datetime.now().time())
            avgTrial = np.mean(im_pat, axis=1) #avg across trials
            avgPrestim = np.mean(avgTrial[br[0]:br[1], :, :, :], axis=(0,3))
            avgPrestim_bc = np.reshape(avgPrestim, (1, avgPrestim.shape[0], avgPrestim.shape[1])) #broadcast extra dim
            avgPoststim = np.mean(avgTrial[br[1]+nHoloFr-RespFr:br[1]+nHoloFr, :, :, :], axis=0)

            print("\nBegan computing min value at ", datetime.now().time(), end=' ')
            absMin = np.min(avgPrestim);print(absMin)#don't include frames that have an artifact
            print("\nEnded computing min value at ", datetime.now().time(), end=' ')
            
            for stim in range(holoPatterns):
                df_2P[:, :, stim] = (avgPoststim[:, :, stim] - absMin) - (avgPrestim - absMin)
                dfoF_2P[:, :, stim] = df_2P[:, :, stim] / (avgPrestim - absMin+1)*100
                dfoFsmooth_2P[:, :, stim] = df_2P[:, :, stim] / (gaussian_filter(avgPrestim, sigma=20) - absMin)*100

                dfoFsmoothtrial_2P[:, :, :, stim] = ((avgTrial[:, :, :, stim] - absMin) - (avgPrestim_bc - absMin)) / (gaussian_filter(avgPrestim, sigma=20) - absMin)*100

        else: #if one stim pattern, proceed
            print('one pattern')
            df_2P = np.squeeze(df_2P); dfoF_2P = np.squeeze(dfoF_2P)
            dfoFsmooth_2P = np.squeeze(dfoFsmooth_2P); dfoFsmoothtrial_2P = np.squeeze(dfoFsmoothtrial_2P)
            im_pat = np.reshape(im, (nFrTrial, nTrial2P, frameSz, frameSz), order='F') #frames/trial, trials, pixels
            avgTrial = np.mean(im_pat, axis=1) #avg across trials
            avgPrestim = np.mean(avgTrial[br[0]:br[1], :, :], axis=0)
            avgPoststim = np.mean(avgTrial[br[1]+nHoloFr-RespFr:br[1]+nHoloFr, :, :], axis=0)
            
            print("\nBegan computing min value at ", datetime.now().time(), end=' ')
            absMin = np.min(avgPrestim); print('min=',absMin)#don't include frames that have an artifact
            
            df_2P = (avgPoststim - absMin) - (avgPrestim - absMin)
            dfoF_2P = df_2P / (avgPrestim - absMin+1)*100
            dfoFsmooth_2P = df_2P / (gaussian_filter(avgPrestim, sigma=20) - absMin)*100
            
            dfoFsmoothtrial_2P = ((avgTrial - absMin) - (avgPrestim - absMin)) / (gaussian_filter(avgPrestim, sigma=20) - absMin)*100


        print('Frame-based dF/Fs are created.')

        return avgPrestim,dfoF_2P,df_2P,dfoFsmooth_2P,dfoFsmoothtrial_2P,im_pat   

    def generate_trialavg_dfof_frame_indivBL(self,im,stimOrder,RespFr=5):
        
        '''generate the trial-averaged dF/F frames for a given image stack
        titles: array of strings, length = patterns, will be assigned a plot'''
        
        print('Generating frame-based trial-averaged dF/F...')

        frameSz = self.frameSz
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial
        br = self.baseline_range
        nHoloFr = self.nHoloFr
        nCell = self.nCell
        holoPatterns = self.holoPatterns
        
        df_2P = np.full((frameSz,frameSz,holoPatterns),np.nan)
        dfoF_2P = np.full(df_2P.shape, np.nan)
        dfoFsmooth_2P = np.full(df_2P.shape, np.nan)
        dfoFsmoothtrial_2P = np.full((nFrTrial,frameSz,frameSz,holoPatterns), np.nan)

        im = np.reshape(im, (nFrTrial, nTrial2P, frameSz, frameSz), order='F')
        im_pat = np.zeros((im.shape[0], int(im.shape[1]/holoPatterns), im.shape[2], im.shape[3], holoPatterns)) #trialFrames, trials_per_pattern, framesz, framesz, patterns 

        for stims in range(len(stimOrder)):
            #which iteration of the pattern are you on? If 80 trials per pattern, this will go to 80
            trialForRandom = math.floor(stims/holoPatterns)
            if stims % 50 == 0:
                print('\nTrial number... ' ,stims, ' ',datetime.now().time(), end=' ')
            im_pat[:, trialForRandom, :, :, stimOrder[stims]-1] = im[:, stims, :, :] #frame, trial, pixel, pixel
        print("\nFinished im_pat ", datetime.now().time())
        
        avgTrial = np.mean(im_pat, axis=1) #avg across trials
        avgPrestim = np.mean(avgTrial[br[0]:br[1], :, :, :], axis=0)
        avgPoststim = np.mean(avgTrial[br[1]+nHoloFr-RespFr+1:br[1]+nHoloFr+1, :, :, :], axis=0)
        print(avgTrial.shape)
        print(avgPrestim.shape)
        print("\nBegan computing min value at ", datetime.now().time(), end=' ')
        absMin = np.min(avgPrestim);print(absMin)#don't include frames that have an artifact
        print("\nEnded computing min value at ", datetime.now().time(), end=' ')

        #for stim in range(holoPatterns):
        df_2P = (avgPoststim - absMin) - (avgPrestim - absMin)
        dfoF_2P = df_2P / (avgPrestim - absMin+0.1)*100
        dfoFsmooth_2P = df_2P / (gaussian_filter(avgPrestim, sigma=10) - absMin)*100

        avgPrestim_bc = avgPrestim.reshape(1, avgPrestim.shape[0], avgPrestim.shape[1], avgPrestim.shape[2]) #broadcasting so dims=avgTrial
        dfoFsmoothtrial_2P = ((avgTrial - absMin) - (avgPrestim_bc - absMin)) / (gaussian_filter(avgPrestim_bc, sigma=10) - absMin)*100

        print('Frame-based dF/Fs are created.')

        return avgPrestim,dfoF_2P,df_2P,dfoFsmooth_2P,dfoFsmoothtrial_2P,im_pat   

    
    
    
    def plot_trialavg_dfof_frame_ngf(self,dfoF_2P,titles,color_lim=30,save_ims=False):
        '''generate the trial-averaged dF/F frames for a given image stack
        titles: array of strings, length = patterns, will be assigned a plot'''
        
        print('Plotting frame-based trial-averaged dF/F...')
        
        holoPatterns = self.holoPatterns

        for stim in range(0, holoPatterns):
            plt.figure()
            save_name = str(stim+1)
            if dfoF_2P.shape == 3:
                pat_dfof = dfoF_2P[:, :, stim]
            else: 
                pat_dfof = dfoF_2P
            pat_dfof = gaussian_filter(pat_dfof, sigma=0.5)
            plt.imshow(pat_dfof, cmap='RdBu_r',vmin=-color_lim,vmax=color_lim)
            plt.title(titles[stim], fontsize = 14)
            plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
            plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
            plt.xticks([i*32 for i in range(9)],[]) #no tick labels
            plt.yticks([i*32 for i in range(9)],[]) #no tick labels
            plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
            plt.colorbar()

            if save_ims == True:
                save_path = os.path.join(self.save_file_path,'2pstim_frame_Pat'+save_name+'.png')
                plt.savefig(save_path,bbox_inches='tight',dpi=400)
                print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)

        return   

    def plot_trialavg_dfof_frame_wpatcoords (self, dfoFsmooth_2P,cell_masks,titles,file_path_pattern,pat_names,whichPtsInPat,save_ims=False,color_lim=30,radius_disk=0,radius_capture=0,downscale_factor=2):
    
        print('Making map with pattern overlayed. Make sure the pattern filename is a string.')

        nCell = self.nCell
        holoPatterns = self.holoPatterns
        zoom = self.zoom
        
        cmap='RdBu_r'
        
        for stim in range(0, holoPatterns):
            fig, ax = plt.subplots()
            save_name = str(stim+1)
            pat_dfof = dfoFsmooth_2P[:, :, stim]
            pat_dfof = gaussian_filter(pat_dfof, sigma=0.5)
            ax.imshow(pat_dfof, cmap=cmap,vmin=-color_lim,vmax=color_lim)
            ax.set_title(titles[stim], fontsize = 14)
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
            cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=14)  # vertically oriented colorbar
            cbar.set_label('df/f',rotation=270,fontsize=14)
            ax.axis('off')
            
            disks = True #not using disks 
            if disks==True:
                cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[stim])
                for coords in pattern_coords:
                    #ax.add_patch( Circle((coords[0],coords[1]),radius=radius_disk*conversion_factor,color='cyan',alpha=0.1) )
                    ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='tab:cyan',ls='--',lw=2,alpha=1) )
                save_name = save_name + 'patcoordsQE'

            dlp = False #using diffraction limited points
            if dlp ==True:
                cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[0])
                first_pt = whichPtsInPat[stim,0]
                last_pt = whichPtsInPat[stim,1]
                for coords in pattern_coords[first_pt:last_pt,:]:
                    #ax.add_patch( Circle((coords[0],coords[1]),radius=radius_disk*conversion_factor,color='cyan',alpha=0.1) )
                    ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='mediumorchid',ls='--',lw=1,alpha=1) )
                save_name = save_name + 'patcoords'

            if save_ims == True:
                save_path = os.path.join(self.save_file_path,'2pstimd_frame'+save_name+'.png')
                plt.savefig(save_path,bbox_inches='tight',dpi=400)
                print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)
        
        return
    
    def subtract_nonlin_nopat (self, dfoFsmooth_2P,subtract_titles,A,B,C,AandB,AandC,save_ims=False,color_lim=20,sigma=0.0):
    
        print('Making map with pattern overlayed. Make sure the pattern filename is a string.')

        nCell = self.nCell
        holoPatterns = self.holoPatterns
        zoom = self.zoom
        cmap='RdBu_r'
        #LaFosseMap
        #from matplotlib.colors import LinearSegmentedColormap
        #cmap = LinearSegmentedColormap.from_list(name='mycmap',colors=[(0,'#40004b'),(0.1,'#ad8abd'),(0.5,'white'),(0.9, '#80c481'),(1,'#01732e')])
        #cmap='PRGn'
        
        pat_dfof_subtract = np.full(dfoFsmooth_2P[:, :, :2].shape, np.nan)
        pat_dfof_subtract[:, :, 0] = dfoFsmooth_2P[:, :, AandB] - (dfoFsmooth_2P[:, :, A] + dfoFsmooth_2P[:, :, B])
        pat_dfof_subtract[:, :, 1] = dfoFsmooth_2P[:, :, AandC] - (dfoFsmooth_2P[:, :, A] + dfoFsmooth_2P[:, :, C])
        pat_dfof_subtract[:, :, 0] = gaussian_filter(pat_dfof_subtract[:, :, 0], sigma=sigma)
        pat_dfof_subtract[:, :, 1] = gaussian_filter(pat_dfof_subtract[:, :, 1], sigma=sigma)

        
        ##############################################################################################
        fig, ax = plt.subplots()
        save_name = 'Cotuned'
        ax.imshow(pat_dfof_subtract[:, :, 0], cmap=cmap,vmin=-color_lim,vmax=color_lim)
        ax.set_title(subtract_titles[0], fontsize = 12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=12)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)

        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.xticks([i*32 for i in range(9)],[]) #no tick labels
        plt.yticks([i*32 for i in range(9)],[]) #no tick labels
        plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
        #ax.axis('off') 
            
        if save_ims == True:
            save_path = os.path.join(self.save_file_path,'subtract_frameSmooth'+save_name+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)
        ##############################################################################################
        
        fig, ax = plt.subplots()
        save_name = 'Antituned'
        ax.imshow(pat_dfof_subtract[:, :, 1], cmap=cmap,vmin=-color_lim,vmax=color_lim)
        ax.set_title(subtract_titles[1], fontsize = 12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=12)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)

        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.xticks([i*32 for i in range(9)],[]) #no tick labels
        plt.yticks([i*32 for i in range(9)],[]) #no tick labels
        plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
        #ax.axis('off')
            
        if save_ims == True:
            save_path = os.path.join(self.save_file_path,'subtract_frameSmooth'+save_name+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)


        return pat_dfof_subtract
    
    def subtract_nonlin (self, dfoFsmooth_2P,cell_masks,subtract_titles,file_path_pattern,pat_names,whichPtsInPat,A,B,C,AandB,AandC,save_ims=False,color_lim=20,radius_disk=0,radius_capture=0,sigma=0.5,downscale_factor=2):
    
        print('Making map with pattern overlayed. Make sure the pattern filename is a string.')

        nCell = self.nCell
        holoPatterns = self.holoPatterns
        zoom = self.zoom
        cmap='RdBu_r'
        
        pat_dfof_subtract = np.full(dfoFsmooth_2P[:, :, :2].shape, np.nan)
        pat_dfof_subtract[:, :, 0] = dfoFsmooth_2P[:, :, AandB] - (dfoFsmooth_2P[:, :, A] + dfoFsmooth_2P[:, :, B])
        pat_dfof_subtract[:, :, 1] = dfoFsmooth_2P[:, :, AandC] - (dfoFsmooth_2P[:, :, A] + dfoFsmooth_2P[:, :, C])
        pat_dfof_subtract[:, :, 0] = gaussian_filter(pat_dfof_subtract[:, :, 0], sigma=sigma)
        pat_dfof_subtract[:, :, 1] = gaussian_filter(pat_dfof_subtract[:, :, 1], sigma=sigma)

        
        ##############################################################################################
        fig, ax = plt.subplots()
        save_name = 'Cotuned'
        ax.imshow(pat_dfof_subtract[:, :, 0], cmap=cmap,vmin=-color_lim,vmax=color_lim)
        ax.set_title(subtract_titles[0], fontsize = 12)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=12)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)

        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.xticks([i*32 for i in range(9)],[]) #no tick labels
        plt.yticks([i*32 for i in range(9)],[]) #no tick labels
        plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
        #ax.axis('off') 
        
        dlp = True #using diffraction limited points
        if dlp==True:
            cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[0])
            first_pt = whichPtsInPat[AandB,0]
            last_pt = whichPtsInPat[AandB,1]
            print(first_pt)
            for coords in pattern_coords[first_pt:last_pt,:]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='mediumorchid',ls='--',lw=1,alpha=1) )
            save_name = save_name + 'patcoords'
            
        if save_ims == True:
            save_path = os.path.join(self.save_file_path,'subtract_frameWcoords'+save_name+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)
        ##############################################################################################
        
        fig, ax = plt.subplots()
        save_name = 'Antituned'
        ax.imshow(pat_dfof_subtract[:, :, 1], cmap=cmap,vmin=-color_lim,vmax=color_lim)
        ax.set_title(subtract_titles[1], fontsize = 12)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=12)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)

        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
        plt.xticks([i*32 for i in range(9)],[]) #no tick labels
        plt.yticks([i*32 for i in range(9)],[]) #no tick labels
        plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
        #ax.axis('off')
        
        dlp = True #using diffraction limited points
        if dlp==True:
            cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[0])
            first_pt = whichPtsInPat[AandC,0]
            last_pt = whichPtsInPat[AandC,1]
            for coords in pattern_coords[first_pt:last_pt,:]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='mediumorchid',ls='--',lw=1,alpha=1) )
            save_name = save_name + 'patcoords'
            
        if save_ims == True:
            save_path = os.path.join(self.save_file_path,'subtract_frameWcoords'+save_name+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)


        return pat_dfof_subtract
    
    def plot_pixelframe_circle_cell (self,dfoFsmooth_2P,cell_masks,cell_iD,titles,file_path_pattern,pat_names,save_ims=False,color_lim=30,radius_disk=10,radius_capture=12,downscale_factor=2):

        print('Making map with pattern overlayed. Make sure the pattern filename is a string.')

        nCell = self.nCell
        holoPatterns = self.holoPatterns
        zoom = self.zoom
        cmap='RdBu_r'

        for i in range(len(cell_iD)):
            for stim in range(0, holoPatterns):

                cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[stim])
                choose_cell = cell_coords[cell_iD[i], :]
                radius_pix = radius_disk*conversion_factor

                for iP in range(pattern_coords.shape[0]):
                    pattern_coord = pattern_coords[iP,:]
                    euclidean_distances = np.linalg.norm(choose_cell-pattern_coord)
                    if euclidean_distances<radius_pix:
                        save_name = 'pattern'+str(stim+1)+'_cell'+str(cell_iD[i])
                        pat_dfof = dfoFsmooth_2P[:, :, stim]

                        fig, ax = plt.subplots()
                        ax.imshow(pat_dfof, cmap=cmap,vmin=-color_lim,vmax=color_lim)
                        ax.set_title(titles[stim], fontsize = 18)
                        ax.add_patch( Circle((pattern_coords[iP, 0],pattern_coords[iP, 1]),radius=radius_disk*conversion_factor,color='cyan',alpha=0.1) )
                        ax.add_patch( Circle((pattern_coords[iP, 0],pattern_coords[iP, 1]),radius=radius_capture*conversion_factor,fill=False,ec='cyan',ls='--',lw=1,alpha=1) )
                        ax.axis('off')
                        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1])
                        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=18)  # vertically oriented colorbar
                        cbar.set_label('df/f',rotation=270)

                        if save_ims == True:
                            cell_path = os.path.join(self.save_file_path,'cells')
                            if not os.path.exists(cell_path):
                                os.makedirs(cell_path, exist_ok=True) 
                            save_path = os.path.join(cell_path,'2pstimd_frame'+save_name+'.png')
                            plt.savefig(save_path,bbox_inches='tight',dpi=400)
                            print('Response map completed. Figure saved at '+save_path)

        return
    
    def cell_characterize_subplot (self,dfoF_trace_lowessV,dfoFsmooth_2P,cell_masks,cell_iD,titles,file_path_pattern,pat_names,color_lim=30,RespFrames=10,radius_disk=10,radius_capture=12,downscale_factor=2,save_ims=False):
        
        nCell = self.nCell
        holoPatterns = self.holoPatterns
        zoom = self.zoom
        nPreFrV = self.nPreFrV
        nStimFrV = self.nStimFrV
        cmap='RdBu_r'

        #get single df/f values for each cell using an average over end of stim frames and lowessed cell mask trace
        dfoF_V_avg = dfoF_trace_lowessV[:, :, nPreFrV+nStimFrV-RespFrames:nPreFrV+nStimFrV].mean(axis=2)

        xs = [0,np.pi*0.25,np.pi*0.5,np.pi*0.75,np.pi,np.pi*1.25,np.pi*1.5,np.pi*1.75, np.pi*2]
        dfoF_Vplot = dfoF_V_avg[:, cell_iD]
        dfoF_Vplot = np.concatenate((dfoF_Vplot, dfoF_Vplot[:1, :]))

        fig=plt.figure(figsize=(14,8))
        fig.text(0.5, 1.02, 'cell '+str(cell_iD[0]), ha='center', va='center', fontsize=30, fontweight='bold')
        font = {'size'   : 18}
        mpl.rc('font', **font)
        save_name = str(cell_iD[0])

        ax1 = fig.add_subplot(231, projection='polar')
        ax1.plot(xs,dfoF_Vplot,color='blue')
        ax1.set_rmax(30)
        ax1.set_rticks([10, 20, 30])  # Less radial ticks
        ax1.set_xlabel('df/f')
        ax1.set_rlabel_position(-20)  # Move radial labels away from plotted line
        ax1.grid(True)
        ax_num = 1

        for stim in range(0, holoPatterns):

                cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[stim])
                choose_cell = cell_coords[cell_iD, :]
                radius_pix = radius_disk*conversion_factor

                for iP in range(pattern_coords.shape[0]):
                    pattern_coord = pattern_coords[iP,:]
                    euclidean_distances = np.linalg.norm(choose_cell-pattern_coord)
                    if euclidean_distances<radius_pix:
                        pat_dfof = dfoFsmooth_2P[:, :, stim]
                        ax[ax_num-1] = fig.add_subplot(231+ax_num)
                        ax[ax_num-1].imshow(pat_dfof, cmap=cmap,vmin=-color_lim,vmax=color_lim)
                        ax[ax_num-1].set_title(titles[stim], fontsize = 18)
                        ax[ax_num-1].add_patch( Circle((pattern_coords[iP, 0],pattern_coords[iP, 1]),radius=radius_disk*conversion_factor,color='cyan',alpha=0.1) )
                        ax[ax_num-1].add_patch( Circle((pattern_coords[iP, 0],pattern_coords[iP, 1]),radius=radius_capture*conversion_factor,fill=False,ec='cyan',ls='--',lw=1,alpha=1) )
                        ax[ax_num-1].axis('off')
                        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[ax_num-1], ticks=[0, 1])
                        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=18)  # vertically oriented colorbar
                        cbar.set_label('df/f',rotation=270)
                        ax_num = ax_num + 1

        plt.tight_layout()
        plt.show()

        if save_ims == True:
            cell_characterize_path = os.path.join(save_file_path,'cell_characteristic_plots')
            if not os.path.exists(cell_characterize_path):
                os.makedirs(cell_characterize_path, exist_ok=True)
            save_path = os.path.join(cell_characterize_path,'characterize_cell'+save_name+'.png')
            plt.savefig(save_path,bbox_inches='tight',dpi=400)
            print('Response map completed. Figure saved at '+save_path)
        return

    def twoPMap(dfoF_2P, avgPrestim, vminbg, divMaxbg, divMin, divMax):

        N = 10
        colors = 100
        vals0 = np.ones((N, 4))

        #hue
        vals0[:10, 0] = 178/256
        vals0[:10, 1] = 34/256
        vals0[:10, 2] = 34/256

        for i in range(0, N, 10):    
            vals0[i:i+10, 3] = np.linspace(0, 1, 10) #non dynamic... oh well.. fix 10

        newcmp0 = mpl.colors.ListedColormap(vals0)

        # variables to edit: plotting variables
        divMinbg = 10 # default = 50; make smaller to lighten background (ex. 300)
        divMaxbg = 7 #default = 2 (make smaller (~1.3) if you want to lighten the background)

        divMin = 4 #default = 50 (make bigger if you want weakly tuned cells to show)
        divMax = 20 #default = 1.5 (make smaller (~1.1) if you want weakly tuned cells to show

        #create images arrays
        if len(dfoF_2P.shape) == 3:
            mask_2P = dfoF_2P[:, :, AandB]
            #mask_2P = np.amax(dfoF_2P, axis=2)
            bg_2P = np.nanmean(avgPrestim, axis=2)
        else: #if one stim type, show that stim
            mask_2P = np.copy(dfoverF_2P)
            bg_2P = np.copy(avgPrestim)

        mask_2P = gaussian_filter(mask_2P, sigma=1)

        #plotting variables for stimulus
        rangeStim = np.nanmax(mask_2P)-np.nanmin(mask_2P);
        print(rangeStim)
        vminStim = np.nanmin(mask_2P) + (rangeStim/divMin); #minimum value displayed in colormap
        vmaxStim = 40 #np.nanmax(mask_2P) - (rangeStim/divMax); #max value displayed in colormap

        #plotting variables for background
        rangebg = np.nanmax(bg_2P) - np.nanmin(bg_2P)
        vminbg = np.nanmin(bg_2P) - (rangebg/divMinbg)
        vmaxbg = np.nanmax(bg_2P) - (rangebg/divMaxbg); #max value displayed in background
        print(np.nanmin(bg_2P))
        print(np.nanmax(bg_2P))

        #plot               
        fig, ax = plt.subplots(figsize = (12, 10))

        ax.imshow(bg_2P, cmap='gray', vmin=vminbg,vmax=vmaxbg)
        ax.imshow(mask_2P, cmap=newcmp0, vmin=vminStim,vmax=vmaxStim)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=newcmp0), ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels([round(vminStim), round(vmaxStim)], fontsize=20)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)
        ax.axis('off')
        #ax.set_title('2P dF/f Map', fontsize=40)
        fig.tight_layout()

        return(plt.savefig('2PwithBackground.png'), mask_2P)  

    
    def make_mov(dfoF_2P, avgTrial, absMin, avgPrestim):
        mov_file_path = os.path.join(file_path,'movies')
        if not os.path.exists(mov_file_path):
            os.makedirs(mov_file_path, exist_ok=True)

        patternNames = ['orth', 'Aandorth', 'A', 'AandB', 'B']
        #patternNames = ['1', '2', '3', '4', '5']

        for pattern in range(0, numStims):
            dfoF_2Pmov = (((avgTrial[:, :, :, pattern] - absMin) - (avgPrestim[:, :, pattern] - absMin)) / (gaussian_filter(avgPrestim[:, :, pattern], sigma=40) - absMin))*100
            filename = 'dfof_movSmooth' + patternNames[pattern] + '.tif'
            tifffile.imsave(os.path.join(mov_file_path, filename), dfoF_2Pmov, bigtiff=True)
        return(dfoF_2Pmov)


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
    levels = np.unique(mwf.stimDf[mwf.levelVar])
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
    nTrialV = expt_params['nTrialV']
    nFrTrial = expt_params['nFrTrial']

    # intialize cell traces reshaped by levels and trials
    traces_parsed = np.zeros((nCell, nLevel, nTrialV, nFrTrial))
    
    # extract values from frames of specified level and trial
    for iC in range(nCell):
        for iL in range(nLevel):
            for iT in range(nTrialV):
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
        nTrialV = expt_params['nTrialV']
        nFrTrial = expt_params['nFrTrial']

        # intialize cell traces reshaped by levels and trials
        im_parsed = np.zeros((nLevel, nTrialV, nFrTrial, im.shape[1], im.shape[2]))
        print(im_parsed.shape)

        # extract values from frames of specified level and trial
        for iL in range(nLevel):
            for iT in range(nTrialV):
                start = (iT*nFrTrial)
                end   = (iT*nFrTrial)+nFrTrial
                im_parsed[iL,iT,:,:,:] = im[levelFr[iL][start:end],:,:]

        return im_parsed
