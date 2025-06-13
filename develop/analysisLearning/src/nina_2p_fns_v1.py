#mworks imports
from mworksbehavior import mwkfiles
from mworksbehavior.imaging import intrinsic as ii
import mworksbehavior as mwb
import mworksbehavior.mwk_io
from glob import glob
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Circle
from PIL import Image

import os, sys
import tifffile
from skimage import transform
from scipy.ndimage import gaussian_filter as gaussian_filter
from scipy import sparse
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import numpy as np
import pandas as pd

import analysis_2p_fxns as p2f

from pathlib import Path
a_ = np.asarray
r_ = np.r_
n_ = np.newaxis

############################

# functions to assist in handling preprocessing of data
def SI_batch_resave(infile, outfile, nFr=-1, nFrChunk=300, rewriteOk=False, downscaleTuple=None):
    """
    :param infile: (str) path to infile
    :param outfile: (str) path to outfile
    :param nFrChunk: (int) number of frames to load into memory at once
    :param rewriteOk: (bool) whether or not to overwrite existing file
    :param downscale: (bool) whether or not to downscale the image
    :param downscaleTuple: (tuple) downscaling factor in (z, x, y), e.g. (1, 2, 2) for 2x downscale
    :return:
    """
    if os.path.isfile(outfile):
        assert rewriteOk, 'outfile already exists but rewriteOK is False'
        os.remove(outfile)
        print('original outfile exists - deleting.')
    with tifffile.TiffFile(infile) as tif:
        if nFr == -1:
            T = len(tif.pages)
        else:
            T = nFr
            assert T <= len(tif.pages), 'number of frames in file is less than user-defined number to process'
        nR,nC = tif.pages[0].shape
    if downscaleTuple is not None:
        nR = int(nR/downscaleTuple[1])
        nC = int(nC/downscaleTuple[2])
    
    im = np.zeros((T,nR,nC))
    for fr in r_[0:T:nFrChunk]:
        if fr+nFrChunk <= T:
            ix = r_[fr:fr+nFrChunk]
            chunk = tifffile.imread(infile,key=ix)
            print(fr+nFrChunk, end=' ')
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        if fr+nFrChunk > T:
            ix = r_[fr:T]
            chunk = tifffile.imread(infile, key=ix)
            print(T)
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        del chunk

    im = im.astype('int16')
    tifffile.imsave(outfile, im, bigtiff=True)
    print('done. saved to {}'.format(outfile))
    

def makePositive(im):
    if np.min(im) < 0:
        im += abs(np.min(im))
        print('Min was below zero. Now corrected.')
    return im

###########################################
class analysis_general:
    
    def __init__(self,file_path,vis_expt_params):
        '''initialize class'''
        save_file_path = os.path.join(file_path,'analysisGeneral')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path, exist_ok=True)
        
        self.save_file_path = save_file_path
        self.expt_params = vis_expt_params
        self.fps = vis_expt_params['fps']
        self.zoom = vis_expt_params['zoom']
        self.nOrientations = vis_expt_params['nOrientations']
        self.nTrialV = vis_expt_params['nTrialV']
        self.nFrTrial = vis_expt_params['nFrTrial']
        self.strtPreFr = vis_expt_params['strtPreFr']
        self.endPreFr = vis_expt_params['endPreFr'] 
        self.stimStrt = vis_expt_params['stimStrt']
        self.stimEnd = vis_expt_params['stimEnd']
        self.nPostFr = vis_expt_params['nPostFr']
        self.nCell = vis_expt_params['nCell']
        
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type,exc_value,traceback):
        return
    
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
        #try:
        mwf = mwkfiles.RetinotopyMap2StimMWKFile(dd2.h5name,stims_to_keep=useStimRange)
        #except:
        #    mwf = mwkfiles.RetinotopyMap1MWKFile(dd2.h5name,stims_to_keep=useStimRange)
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
    
    def calculate_cell_pattern_coords(self,file_path_pattern,cell_masks,csv_pattern_filename='pattern1.csv',downscale_factor=2):
        '''calculate coordinates of cell mask center of masses and coordinates of stim pattern targets'''
        # get coordinates of center of masses of masks in pixels
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
    

    def dfoF_orient(self,im_stimV,mwf,expt_params,smoothVal):
        
        # parse mwk level information
        levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        
        # define experiment parameters
        trials = expt_params['nTrialV']
        trialFrames = expt_params['nFrTrial']
        prestimStrt = expt_params['strtPreFr']
        endPreFr = expt_params['endPreFr']
        stimStrt = expt_params['stimStrt']
        stimEnd = expt_params['stimEnd']

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
        
    def plot_fullFrame(self,dfoF_vis,mwf,expt_params,save=False,sigma=0.5,color_lim=20):
        
        # define experiment parameters
        #trials = expt_params['nTrialV']
        #trialFrames = expt_params['nFrTrial']
        #prestimStrt = expt_params['strtPreFr']
        #endPreFr = expt_params['endPreFr']
        #stimStrt = expt_params['stimStrt']
        #stimEnd = expt_params['stimEnd']
        nOrientations = self.nOrientations
        
        #levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        #levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#) 60preframes, 30stim, 210post
        if nOrientations == 3:
            oriDegrees = ['240', '280', '320']
        if nOrientations == 4:
            oriDegrees = ['0', '45', '90','135']
        if nOrientations == 8:
            oriDegrees = ['0','45','90','135','180','225','270','315']
        if nOrientations == 9:
            print('new order established now 16 2022 when I switched the variables in mworks. old is 0;-15,15;-15,30;-15,0;0,15;0,30;0,0;15,15;15, 30;15]')
            oriDegrees = ['0;15','15;15','30;15','0;0','15;0','30;0','0;-15','15;-15', '30;-15']
        
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
    
    def plot_fullFrame_withpatCoords (self,dfoF_vis,mwf,expt_params,cell_masks,file_path_pattern,pat_names,pat_show,whichPtsInPat,save=False,sigma=0.5,color_lim=20,radius_disk=0,radius_capture=0,downscale_factor=2):
        
        # define experiment parameters
        nOrientations = self.nOrientations
        zoom = self.zoom
        trials = self.nTrialV
        trialFrames = self.nFrTrial
        cmap='RdBu_r'
        pattern_colors = ['mediumorchid','mediumslateblue','lightblue']

        levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#)

        if nOrientations == 4:
            oriDegrees = ['0', '45', '90','135']
        if nOrientations == 8:
            oriDegrees = ['0','45','90','135','180','225','270','315']

        for ori in range(0, nLevels):
            fig, ax = plt.subplots(figsize=(12,9))
            pat_dfof = dfoF_vis[ori, :, :]
            ax.imshow(gaussian_filter(pat_dfof,sigma=sigma), cmap=cmap,vmin=-color_lim,vmax=color_lim)
            ax.set_title(oriDegrees[ori]+'°', fontsize = 20)
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=20)  # vertically oriented colorbar
            cbar.set_label('df/f',rotation=270)
            ax.axis('off')
            
            for pats in range(pat_show.shape[0]):
                cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[pats])
                print(pat_names[pats])
                for coords in pattern_coords:
                    ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='mediumorchid',ls='--',lw=2,alpha=1) )

#below if you wanna just put patterns on stim'd maps                    
#             if ori == 0 or ori == 1 or ori == 2 or ori == 3:
#                 for pats in range(pat_show.shape[0]):
#                     cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[pats])
#                     first_pt = whichPtsInPat[0,0]
#                     last_pt = whichPtsInPat[0,1]
#                     for coords in pattern_coords[first_pt:last_pt,:]:
#                         ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec=pattern_colors[0],ls='--',lw=1,alpha=1) )
            
            # if ori < 10:
            #     for pats in range(pat_show.shape[0]):
            #         cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[pats])
            #         first_pt = whichPtsInPat[4,0]
            #         last_pt = whichPtsInPat[4,1]
            #         for coords in pattern_coords[first_pt:last_pt,:]:
            #             ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec=pattern_colors[0],ls='--',lw=1,alpha=1) )

            if save:
                save_path = os.path.join(self.save_file_path,'trialavg_Ori_fullFramewithPat'+str(ori+1)+'.png')
                plt.savefig(save_path,bbox_inches='tight')
                print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)

        return pat_dfof
    
    def onePlot_allOris(self,dfoF_vis,expt_params,save=False,sigma=0.3,color_lim=20):
        
        nOrientations = self.nOrientations
        
        try:
            nOrientations == 8
        except ValueError:
            print("Error: Fuction is set up specifically for 8 orientations")

        titles = ['0°', '45°', '90°', '135°', '180°', '225°', '270°','315°']
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize = (20, 7.7))
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.1)

        ax = ax.ravel()

        for stim in range(len(titles)):
            pat_dfof = dfoF_vis[stim, :, :]
            ax[stim].imshow(pat_dfof, cmap='RdBu_r',clim=r_[-color_lim,color_lim])
            ax[stim].set_title(titles[stim], fontsize = 14)
            ax[stim].grid(True)#(which='major', axis='both')
            ax[stim].set_xticklabels([])
            ax[stim].set_yticklabels([])
            ax[stim].tick_params(length=0, grid_alpha=0.4)

        cbar = fig.colorbar(cm.ScalarMappable(cmap='RdBu_r'), ax=ax, ticks=[0, 0.5, 1],shrink=0.99, pad=0.01)
        cbar.ax.set_yticklabels([-color_lim, 0, color_lim], fontsize=20)# vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)
        
        if save:
            save_path = os.path.join(self.save_file_path,'AllOris_fullFrame.png')
            plt.savefig(save_path,bbox_inches='tight')
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)
                
        return

    
    def plot_retinoFrames(self,dfoF_vis,expt_params,save=False,sigma=0.5,color_lim=20):
        
        # define experiment parameters
        nOrientations = self.nOrientations
                
        if nOrientations == 9:
            print('new order established now 16 2022 when I switched the variables in mworks. old is 0;-15,15;-15,30;-15,0;0,15;0,30;0,0;15,15;15, 30;15]')
            oriDegrees = ['0;15','15;15','30;15','0;0','15;0','30;0','0;-15','15;-15', '30;-15']
        #stim_list = [6, 7, 8, 3, 4, 5, 0, 1, 2]
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize = (14, 12.5))
        plots = plt.gca()

        ax = ax.ravel()
        cmap = 'RdBu_r'
        for stim in range(dfoF_vis.shape[0]):
            ax[stim].imshow(gaussian_filter(dfoF_vis[stim,:,:],sigma=sigma),cmap=cmap,vmin=-color_lim,vmax=color_lim)
            ax[stim].set_title(oriDegrees[stim]+'°', fontsize=16)
            ax[stim].set_xticks([]);     ax[stim].set_yticks([])
        plt.subplots_adjust(wspace=0.0, hspace=0.15)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], shrink=0.95, pad=0.02)
        cbar.ax.set_yticklabels([-color_lim, color_lim], fontsize=20)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)

        if save:
            save_path = os.path.join(self.save_file_path,'trialavg_fullFrameRetino.png')
            plt.savefig(save_path,bbox_inches='tight', dpi=400)
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)

        return
    
    def plot_maxRespFrame_withpatCoords (self,dfoF_vis,mwf,expt_params,cell_masks,file_path_pattern,pat_names,pat_show,save=False,color_lim=20,radius_disk=0,radius_capture=0,downscale_factor=2):
        
        # define experiment parameters
        nOrientations = self.nOrientations
        zoom = self.zoom
        trials = self.nTrialV
        trialFrames = self.nFrTrial
        cmap='Reds'
        pattern_colors = ['orchid','mediumslateblue','sandybrown']

        levels,levelFr,nLevels = self.parse_mwk_levels(mwf) #######new code
        levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#)

        fig, ax = plt.subplots(figsize=(12,9))
        pat_dfof = np.amax(abs(dfoF_vis), axis=0)
        ax.imshow(pat_dfof, cmap=cmap,vmin=0,vmax=color_lim)
        ax.set_title('Max response to drifting gratings', fontsize = 20)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels([0, color_lim], fontsize=20)  # vertically oriented colorbar
        cbar.set_label('df/f',rotation=270)
        ax.axis('off')

        for pats in range(pat_show.shape[0]):
            cell_coords,pattern_coords,conversion_factor = self.calculate_cell_pattern_coords(file_path_pattern,cell_masks,csv_pattern_filename=pat_names[pats])
            for coords in pattern_coords:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_disk*conversion_factor,color=pattern_colors[pats],alpha=0.1) )
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec=pattern_colors[pats],ls='--',lw=1,alpha=1) )

        if save:
            save_path = os.path.join(self.save_file_path,'trialavg_maxOri_fullFramewithPat.png')
            plt.savefig(save_path,bbox_inches='tight')
            print('Trial-averaged dF/F response map completed. Figure saved at '+save_path)

        return pat_dfof
    
    def plot_tuningcurve_cell (self,cell_iD,dfoF_trace_lowessV):

        stimStrt = self.stimStrt
        stimEnd = self.stimEnd

        #get single df/f values for each cell using an average over end of stim frames and lowessed cell mask trace
        dfoF_V_avg = dfoF_trace_lowessV[:, :, stimStrt:stimEnd].mean(axis=2)

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
    
    def generate_cell_masks_suite2p(self,file_path_masks,xDim=256,yDim=256):
        '''pulls cell component masks from suite2p output; returns masks as shape (yDim,xDim,nCell)'''
        print('Getting masks.')
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
    
    def generate_dfof_cell_traces(self,im,cell_masks,mwf):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''
        # set dF/F parameters
        trials = self.nTrialV
        trialFrames = self.nFrTrial
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
        # use cell_masks to define raw calcium trace for a given cell region (average across masked region)
        raw_traces = self._im_mask_and_avg(im,cell_masks) # returns as shape (nCell,nFrames)
        print(raw_traces.shape)
        
        if im_bg > 4:
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
    
    def generate_dfof_cell_traces_indivBL(self,im,cell_masks,mwf,F_s2p=np.array([]),use_suite2P_trace=False):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''
        # set dF/F parameters
        
        trials = self.nTrialV
        trialFrames = self.nFrTrial
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
    
    def generate_trialavg_deconv(self,mwf,spikeEst_v):
        '''apply cell masks to an image and calculate the average fluorescence, then 
        dF and dF/F values across all frames'''

        # set dF/F parameters
        trials = self.nTrialV
        trialFrames = self.nFrTrial
        nCell = self.nCell
        
        levels,levelFr,nLevels = self.parse_mwk_levels(mwf)
        levelFrDir = np.reshape(levelFr, (nLevels, trials, trialFrames)) # (orientations, trial#, frame#) 60preframes, 30stim, 210post
        print('levelFirDir:',levelFrDir.shape)

        avgTrial_dc = np.zeros(shape = (nLevels, nCell, trialFrames))
        spikeV_reshape = spikeEst_v.reshape(nCell, trials*nLevels, trialFrames)
        dc_V= np.zeros(shape = (nLevels, nCell, trials, trialFrames))

        for ori in range(0, nLevels):
            frame_num = levelFrDir[ori, :, :]#extract all frame # at that orientation
            dc_V[ori, :, :, :] = spikeEst_v[:, [frame_num]].squeeze()# returns shape nCell, trials, trialFrames
    
        avgTrial_dc= np.mean(dc_V, axis=2) #returns shape holos, nCells, trialframes

        return avgTrial_dc, dc_V

    
    def compute_trialavg_2pstim_map_cell(self,im,dfoF_vis_cell,cell_masks,color_lim=50,save=False):
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
      
#################