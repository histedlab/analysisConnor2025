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
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

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
        self.nFrTrial2P = expt_params['nFrTrial2P']
        self.nPreFr = expt_params['nPreFr']
        self.nStimFr = expt_params['nStimFr']
        self.nPostFr2P = expt_params['nPostFr2P']
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
    
    def compute_trialavg_dfof_frame_ngf(self,im,stimOrder,RespFr=5):
        
        '''generate the trial-averaged dF/F frames for a given image stack
        titles: array of strings, length = patterns, will be assigned a plot'''
        
        print('Generating frame-based trial-averaged dF/F...')

        frameSz = self.frameSz
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial2P
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

    def generate_dfof_cell_traces(self,im,cell_masks):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''
        # set dF/F parameters
        nTrial2P = self.nTrial2P
        nFrTrial = self.nFrTrial2P
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
        nFrTrial = self.nFrTrial2P
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
        nFrTrial = self.nFrTrial2P
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

# ### utility functions

# # fxns for dealing with caiman, suite2p, or mworks data

#### general 2p analysis class - handles all the common quick analyses we do    
class analysis_vis_and_other:
    
    def __init__(self,file_path,vis_oth_params):
        '''initialze class'''
        save_file_path = os.path.join(file_path,'analysisGeneral')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path, exist_ok=True)
        
        self.save_file_path = save_file_path
        self.vis_oth_params = vis_oth_params
        self.fps = vis_oth_params['fps']
        self.zoom = vis_oth_params['zoom']
        self.frameSz = vis_oth_params['frameSz']
        self.nOrientations = vis_oth_params['nOrientations']
        self.nTrial2P = vis_oth_params['nTrial2P']
        self.nTrialV = vis_oth_params['nTrialV']
        self.nFrTrialV = vis_oth_params['nFrTrialV']
        self.nFrTrial2P = vis_oth_params['nFrTrial2P']
        self.nPreFr = vis_oth_params['nPreFr']
        self.nStimFr = vis_oth_params['nStimFr']
        self.nPostFrV = vis_oth_params['nPostFrV']
        self.nPostFr2P = vis_oth_params['nPostFr2P']
        self.strtPreFr = vis_oth_params['strtPreFr']
        self.endPreFr = vis_oth_params['endPreFr'] 
        self.stimStrt = vis_oth_params['stimStrt']
        self.stimEnd = vis_oth_params['stimEnd']
        self.baseline_range = vis_oth_params['baseline_frame_range']
        self.nHoloFr = vis_oth_params['nHoloFr']
        self.nCell = vis_oth_params['nCell']
        self.holoPatterns = vis_oth_params['holoPatterns']

        # two parameters relating to masks are saved in the generate_cell_masks_XXXX functions
        # self.file_path_masks - retains the path to the file used to generate the masks from
        # self.cell_masks - retains the actual mask arrays used on this data set, also returned by the above function
        
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type,exc_value,traceback):
        return

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

    def load_mwk_output(self,file_path,stim_range='all'):
        '''load the .mwk file to extract mworks experiment parameters'''
        print('loading...')        
        try:
            mwkfile = glob(file_path+'/*.mwk*')[0]
        except:
            raise OSError('\nNo .mwk file found at specified path. Make sure path is defined correctly.')
        mwkname = os.path.join(file_path,mwkfile)
        # for fixing up corrupt files
        #useStimRange = np.r_[:1]#stim_range
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
    
    def parse_mwk_levels(self,mwf):
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
    
    def load_mwk_output(self,file_path,stim_range='all'):
        '''load the .mwk file to extract mworks experiment parameters'''
        print('loading...')        
        try:
            mwkfile = glob(file_path+'/*.mwk*')[0]
        except:
            raise OSError('\nNo .mwk file found at specified path. Make sure path is defined correctly.')
        mwkname = os.path.join(file_path,mwkfile)
        # for fixing up corrupt files
        #useStimRange = np.r_[:1]#stim_range
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
    
    def dfoF_orient(self,im_stimV,mwf,smoothVal):
        
        # parse mwk level information
        levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        
        # define experiment parameters
        trials = self.nTrialV
        trialFrames = self.nFrTrialV
        prestimStrt = self.strtPreFr
        endPreFr = self.endPreFr
        stimStrt = self.stimStrt
        stimEnd = self.stimEnd

        levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#) 60preframes, 30stim, 210post

        side_length = np.shape(im_stimV)[1]
        frames = np.zeros(shape = (nLevels, trials, trialFrames, side_length, side_length))
        trial_dfoFsmooth = np.zeros(shape = (nLevels, trialFrames, side_length, side_length))
        df_vis = np.zeros(shape = (nLevels, side_length, side_length))
        #dfoFsmooth = np.zeros(shape = (nLevels, side_length, side_length))
        print('Made empty arrays...')
        
        absMin = np.min(np.mean(im_stimV, axis=0))
        print('background: ', absMin)
        
        for ori in range(0, nLevels):
            frame_num = levelFrDir[ori, :, :] #extract all frame # at that orientation
            #grab those frames
            frames[ori, :, :, :, :] = im_stimV[[frame_num], :, :].reshape(trials, trialFrames, side_length, side_length) #reshape removes unneccesary 6th dim (length=1)
        
        frames = frames-absMin
        avgTrial = np.mean(frames, axis=1) #mean across trials (usually 20) avgtrial is [ori, trialLen, sidelen, sidelen]
        avgPrestim = np.mean(avgTrial[:,prestimStrt:endPreFr, :, :], axis=1)
        avgStim = np.mean(avgTrial[:,stimStrt:stimEnd, :, :], axis=1)
        
        df_vis = avgStim - avgPrestim
        dfoF_smooth = df_vis/gaussian_filter(avgPrestim, sigma=smoothVal)*100
        
        avgPrestim_bc = np.reshape(avgPrestim, (avgPrestim.shape[0], 1, avgPrestim.shape[1], avgPrestim.shape[2])) #avgprestim broadcast 
        trial_dfoFsmooth = (avgTrial - avgPrestim_bc) / gaussian_filter(avgPrestim_bc, sigma=10)*100
        
        return frames, df_vis, dfoF_smooth, avgPrestim, absMin, trial_dfoFsmooth
    
    def generate_dfof_cell_traces_indivBL(self,im,cell_masks,mwf,F_s2p=np.array([]),use_suite2P_trace=False):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''
        # set dF/F parameters
        
        trials = self.nTrialV
        trialFrames = self.nFrTrialV
        strtPreFr = self.strtPreFr
        endPreFr = self.endPreFr
        nCell = self.nCell
        stimStrt = self.stimStrt
        stimEnd = self.stimEnd

        # parse mwk level information
        levels,levelFr,nLevels = self.parse_mwk_levels(mwf)
        levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#) 60preframes, 30stim, 210post
        print(levelFrDir.shape)
        print('Applying masks to image and generating cell traces...')

        # find background subtraction value from across pre-stim frames (minimum pixel value)
        aip = np.mean(im, axis=0)
        im_bg = np.min(aip) # shape is orientation,trials,trialframes,sidelength,sidelength
        print('95th percentile = ', np.percentile(aip, 95)) # shape is orientation,trials,trialframes,sidelength,sidelength
        print('minimum = ', im_bg)
        if use_suite2P_trace == False:
            # use cell_masks to define raw calcium trace for a given cell region (average across masked region)
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

        avgTrial = np.zeros(shape = (nLevels, nCell, trialFrames))
        trialavg_dfoF_traceV = np.zeros(avgTrial.shape)
        df_vis_cell = np.zeros(shape = (nLevels, nCell))
        dfoF_vis_cell = np.zeros(df_vis_cell.shape)
        F_ori = np.zeros(shape = (nLevels, nCell, trials, trialFrames))

        # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
        for ori in range(0, nLevels):
            frame_num = levelFrDir[ori, :, :]#extract all frame # at that orientation
            F_ori[ori, :, :, :] = F[:, [frame_num]].squeeze()# returns shape nCell, trials, trialFrames

        avgTrial_F = np.mean(F_ori, axis=2) #returns shape nLevels, nCells, trialframes
        avgPrestim = np.mean(avgTrial_F[:, :, strtPreFr:endPreFr], axis=2) #returns shape nLevels,nCells
        avgStim = np.mean(avgTrial_F[:, :, stimStrt:stimEnd], axis=2) #returns shape nLevels, nCells

        df_vis_cell = avgStim - avgPrestim
        dfoF_vis_cell = df_vis_cell / (avgPrestim)*100
        
        print(avgPrestim.shape)
        
        avgPrestim_bc = avgPrestim.reshape(avgPrestim.shape[0], avgPrestim.shape[1], 1) #broadcasting so it is same ndims as trialavg
        avgPrestim_bc2 = avgPrestim.reshape(avgPrestim.shape[0], avgPrestim.shape[1], 1, 1) #broadcasting so it is same ndims as F_ori
        trialavg_dfoF_traceV =  (avgTrial_F - avgPrestim_bc) / avgPrestim_bc*100
        trial_dfoF_traceV = (F_ori - avgPrestim_bc2) / avgPrestim_bc2*100

        # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
        #dF = np.zeros(F.shape)
        #dFoF = np.zeros(F.shape)
        #for iC in range (nCell):
        #    for iF in range (nTrial2P*nFrTrial):
        #        dF[iC,iF] = F[iC,iF]-baseF[iC]
        #        dFoF[iC,iF] = (dF[iC,iF]/baseF[iC]) * 100

        return avgTrial_F, im_bg, avgPrestim, trialavg_dfoF_traceV, trial_dfoF_traceV, df_vis_cell, dfoF_vis_cell, F_ori
    
    def plot_fullFrame(self,dfoF_vis,mwf,save=False,sigma=0.5,color_lim=20):
        
        # define experiment parameters
        nOrientations = self.nOrientations
        
        #levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        #levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#) 60preframes, 30stim, 210post
        if nOrientations == 4:
            oriDegrees = ['0', '45', '90','135']
        if nOrientations == 8:
            oriDegrees = ['0','45','90','135','180','225','270','315']
        
        cmap = 'RdBu_r'
        for ori in range(nOrientations):
            pat_dfof = dfoF_vis[ori, :, :]
            fig, ax = plt.subplots()
            ax.imshow(gaussian_filter(pat_dfof,sigma=sigma), cmap='RdBu_r',vmin=-color_lim,vmax=color_lim)
            ax.set_title(oriDegrees[ori]+'°', fontsize=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
            cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=16)  # vertically oriented colorbar

            plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
            plt.grid(color='lightgrey',which='major',linestyle='-',linewidth=2,alpha=0.3)
            plt.xticks([i*32 for i in range(9)],[]) #no tick labels
            plt.yticks([i*32 for i in range(9)],[]) #no tick labels
            plt.gca().tick_params(axis=u'both', which=u'both',length=0) #no ticks
            plt.grid(which='major')
            
            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_fullFrame'+str(oriDegrees[ori])+'.png')
                plt.savefig(save_path,bbox_inches='tight', dpi=400)
                print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)
                
        return pat_dfof
    
    def compute_trialavg_vis_map_cell(self,im,dfoF_vis_cell,cell_masks,color_lim=50,save=False):
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
        trials = self.nTrialV
        nOrientations = self.nOrientations
        nRows = im.shape[1]
        nCols = im.shape[2]
        nCell = self.nCell
        dFoF_map_cell_masks_all = np.zeros((nOrientations, nCell,nRows,nCols))
        
        # plot cell mask dF/F map
        cmap = 'RdBu_r'
        v = color_lim
        
        if nOrientations == 4:
            oriDegrees = ['0', '90', '180','270']
        if nOrientations == 8:
            oriDegrees = ['0','45','90','135','180','225','270','315']

        
        for oris in range(nOrientations):
            
            # generate map of avg dF/F for each cell mask
            dFoF_map_cell_masks = np.zeros((nCell,nRows,nCols))
            
            for iC in range(nCell):
                cell_map = dfoF_vis_cell[oris, iC] * cell_masks[:,:,iC]
                cell_map[cell_map==0] = np.nan
                dFoF_map_cell_masks[iC,:,:] = cell_map
            
            dFoF_map_cell_masks_all[oris,:,:,:] = dFoF_map_cell_masks
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
            plt.title('dF/F; %s Reps' % (trials) + oriDegrees[oris] + '°', fontsize=16)
            
            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_cellOri_df'+oriDegrees[oris]+'.png')
                plt.savefig(save_path,bbox_inches='tight')
                print('Trial-averaged dF/F response map for cell masks completed. Figure saved at '+save_path)
            else:
                print('Trial-averaged dF/F response map for cell masks completed.')

        return dFoF_map_cell_masks_all
    
    def smoothWithLowess(self, dfoF_cell_trace):
        '''smooth your trial averaged data with a lowess filter
        '''
        print('Smoothing data with Lowess filter...')
        
        dimTimeseries = len(dfoF_cell_trace.shape)-1 #assumes the final dimension is the dimension of timepoints/frames
        
        dfoF_trace_lowess = np.zeros(dfoF_cell_trace.shape)
        fracDivisor = dfoF_cell_trace.shape[dimTimeseries]/7 # frac = The fraction of the data used when estimating each y-value
        for stim in range(dfoF_cell_trace.shape[0]):
            print('Smoothing stim ' + str(stim))
            for cell in range(dfoF_cell_trace.shape[dimTimeseries-1]):
                tempLowess = lowess(dfoF_cell_trace[stim, cell, :], np.arange(dfoF_cell_trace.shape[dimTimeseries]), frac= 1./fracDivisor)
                dfoF_trace_lowess[stim, cell, :] = tempLowess[:, 1]
                
        return dfoF_trace_lowess  
