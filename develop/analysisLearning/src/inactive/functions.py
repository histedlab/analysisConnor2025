# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Demo notebook on defining stimulated and non-stimulated cells from pattern coordinates

# + tags=[]
# import and load modules
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

import os, sys
import numpy as np
import tifffile as tfl
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D




##----------------------------------------------------------------
def generate_cell_masks_suite2p(file_path_masks,xDim=256,yDim=256):
    '''pulls cell component masks from suite2p output; returns masks as shape (yDim,xDim,nCell)'''
    # get file_names for necessary file data
    suite2p_results_file = os.path.join(file_path_masks, 'stat.npy')
    suite2p_iscell_file = os.path.join(file_path_masks, 'iscell.npy')
    #suite2p_results_file = os.path.join(file_path_masks,'suite2p/plane0','stat.npy')
    #suite2p_iscell_file = os.path.join(file_path_masks,'suite2p/plane0','iscell.npy')
    
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
    return cell_masks,nCell



##----------------------------------------------------------------
def calculate_cell_pattern_coords(file_path_pattern,cell_masks,zoom,nCell,csv_pattern_filename='pattern1.csv',downscale_factor=2,pattern_template_units='micron'):
    '''calculate coordinates of cell mask center of masses and coordinates of stim pattern targets'''
    # get coordinates of center of masses of masks in pixels
    cell_coords = []
    for iC in range(nCell):
        ys,xs = np.where(cell_masks[:,:,iC]>0)
        coord = [np.average(xs),np.average(ys)]
        cell_coords.append(coord)
    cell_coords = np.asarray(cell_coords)
    # get coordinates of pattern targets in microns
    pattern_file = os.path.join(file_path_pattern,csv_pattern_filename)
    df = pd.read_csv(pattern_file)
    coords = np.rollaxis(np.asarray((df['X'],df['Y'])),1,0)
    # convert from microns to pixels (hard-coded numbers based on SI defaults)
    # and scale to match downscaled image
    if pattern_template_units == 'micron':
        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron
    elif pattern_template_units == 'pixel':
        conversion_factor = (1/downscale_factor)
    else:
        raise 'pattern_template_units must be |micron| or |pixel|'
    pix_coords = coords*conversion_factor
    # remove first entry - zero order position
    pattern_coords = pix_coords[1:]
    return cell_coords,pattern_coords



##----------------------------------------------------------------
def label_stimulated_cells(cell_coords,pattern_coords,zoom,nCell,radius_capture_microns=10,downscale_factor=2):
    '''compare coords of pattern targets and cells to find which cells are directly stimulated 
    within a radius of the pattern target coordinates'''
    print('Finding cell IDs within direct stimulation radius...')
    # convert radius_microns to radius_pix
    conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron
    radius_pix = radius_capture_microns*conversion_factor
    # find within-radius cell_coords
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



##----------------------------------------------------------------
def plot_cell_contours_stim_pattern(cell_masks,zoom,stim_iC,pattern_coords=None,radius_disk=0,radius_capture=0,downscale_factor=2):
    conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

    masks = np.sum(cell_masks,axis=2)
    stim_masks = np.sum(cell_masks[:,:,stim_iC],axis=2)
    masks[masks>0] = 1
    masks[masks==0] = 0

    stim_masks[stim_masks>0] = 1
    stim_masks[stim_masks==0] = np.nan

    plt.figure(figsize=np.r_[1,1]*8)
    ax = plt.gca()
    plt.axis('off')

    # plot masks
    plt.imshow(masks,vmin=0,vmax=2,cmap='Greys_r')
    plt.imshow(stim_masks,vmin=0,vmax=1,cmap='viridis')
    
    # plot pattern
    if pattern_coords is not None:
        for coords in pattern_coords:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_disk*conversion_factor,color='cyan',alpha=0.2) )
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='cyan',ls='--',lw=1,alpha=1) )
      
    # for building custom legend
    line = Line2D([],[],color='white',ls='-',linewidth=0,label='Capture Radius')
    sc = plt.scatter([],[],s=14**2,facecolors='white',edgecolors='cyan',linestyle='--',linewidth=2)
    legend_elements = [Line2D([0],[0],markersize=15,marker='o',color='w',markerfacecolor='c',alpha=0.5,label='Disk Pattern'),
                       (line,sc),
                       Line2D([0],[0],markersize=15,marker='o',color='w',markerfacecolor='yellow',label='Stimulated Cell'),
                       Line2D([0],[0],markersize=15,marker='o',color='w',markerfacecolor='gray',label='Non-stimulated Cell')]
    legend_labels = ['Disk Pattern','Capture Radius','Stimulated Cell','Non-stimulated Cell']
    ax.legend(legend_elements,legend_labels,bbox_to_anchor=(1, 1),loc='upper left',fontsize=12)
    
    plt.show()
    
    
    
    
##----------------------------------------------------------------
def generate_dfof_cell_traces(raw_traces,im,trial_params):
    '''apply cell masks to an image and calculate the average fluorescence, then 
    dF and dF/F values across all frames'''
    # set dF/F parameters
    print('im: [trials,frame,x,y] for images in trials')
    nTrial = trial_params['nTrial']
    nTrialF = trial_params['nTrial']
    nFrTrial = trial_params['nFrTrial']
    nPreFr = trial_params['nPreFr']
    nCell = trial_params['nCell']
    br = trial_params['baseline_range']

    # find background subtraction value from across pre-stim frames (minimum pixel value)
    # im_raw = im.reshape((nTrial,nFrTrial,im.shape[2],im.shape[3]))
    im_raw = im.reshape((nTrial,nFrTrial,im.shape[1],im.shape[2]))
    
    im_bg = im_raw[:,:nPreFr,:,:].min()

    # subtract background value from all traces
    F = raw_traces - im_bg
    print('F.shape: ', F.shape)

    # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
    baseF = F.reshape((nCell,nTrialF,nFrTrial))[:,:,br[0]:br[1]].mean(axis=(1,2))
    #baseF = F.reshape((nCell,nTrial,nFrTrial))[:,-1,br[0]:br[1]].mean(axis=(1,2)) #ZZ

    
    # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
    dF = np.zeros(F.shape)
    dFoF = np.zeros(F.shape)
    for iC in range (nCell):
        for iF in range (nTrialF*nFrTrial):
            dF[iC,iF] = F[iC,iF]-baseF[iC]
            dFoF[iC,iF] = (dF[iC,iF]/baseF[iC]) * 100

    return dFoF,dF,F,baseF


##----------------------------------------------------------------
# ##
# ## plot ROIs in image (dF or dF/F) and plot traces
# #### im_file: [frames, 256, 256]
# #### xy_file: 512x512 -> *1.235
# #### xy_file format should be like, no more Area Min Max masurements: 
# ![image.png](attachment:05e78a07-93df-44e0-afeb-7678630b7f56.png)
def dFoF_ROI_plots(im_file, xy_rois, trial_params, analysis_params, savefig_name=0):
    #load images
    print('im_file: average of trails of frames')
    print('xy_rois: xy oocrdinates of ROIs get from ImageJ multi-point')
    
    print('im_file shape:', np.shape(im_file))
    print('xy_rois: ', np.shape(xy_rois))
    n_rois = np.shape(xy_rois)[0]
    
    br = trial_params['baseline_range']
    nPreFr = int(trial_params['nPreFr'])
    nStimFrs_cut = trial_params['nStimFrs_cut']
    nRespFrs = trial_params['nRespFrs']
    roi_size = analysis_params['roi_size']
    print('baseline_range[0]: ', br[0])     
    
    xy_rois_up = (xy_rois-roi_size/2).astype(int)

    
    nr = int(input('nr (cell number in pattern 2) --> 0: only one pattern'))

    #subtract the minimum value as background
    #im_s_reduced = block_reduce(im_file, block_size=(1,8,8), func=np.mean, cval=np.mean(im_file)) #block_size is downsamping ratio? maybe not
    #print('im_s_reduced shape: ', np.shape(im_s_reduced))  
    #F0_min = im_s_reduced[5:(nPreFr-5),2:30,2:30].min()
    F0_min = im_file[5:(nPreFr-5),:,:].min()
    print('F0_min: ', F0_min)
    #im_bg = im_raw[:,:nPreFr,:,:].min()
    
    im_s = im_file - F0_min     
    #print('min: ', im_s_reduced[5:(nPreFrs-2),2:30,2:30].min())
    #print('im_s[br[0]:br[1],:,:].shape: ', np.mean(im_s[br[0]:br[1],:,:],axis=0).shape)

    # remove stim artifact by subtract noise, noise = first_stim_frame(usually fr60) - average_of_baseline_frames
    artifact_remove = int(input('To remove stimulus artifct (Y/N): 1/0'))
    if artifact_remove:
        if nStimFrs_cut > 0:
            noise0 = im_s[nPreFr,:,:] - np.mean(im_s[br[0]:br[1],:,:],axis=0)
            noise_removal = im_s[nPreFr:nPreFr+nStimFrs_cut,:,:] - noise0
            print('stim noise corrected frame: ', np.shape(noise_removal))
            im_s = np.concatenate((im_s[:nPreFr,:,:], noise_removal, im_s[nPreFr+nStimFrs_cut:,:,:]))
    print('np.shape(im_s)', np.shape(im_s))
    
    df = im_s - np.mean(im_s[br[0]:br[1],:,:], axis=0)
    dfof = 100*df/np.mean(im_s[br[0]:br[1],:,:], axis=0)
    
    base = np.mean(im_s[br[0]:br[1],:,:],axis=0) # average for baseline
    resp = np.mean(im_s[nPreFr+nStimFrs_cut:nPreFr+nStimFrs_cut+nRespFrs,:,:],axis=0) # average for response
    
    df_map = resp-base
    dfof_map = ((resp-base))/(base)*100 # dF/F calculation and conversion to percent change
    print('dfof_map.shape', np.shape(dfof_map))
    
    vmax1 = (np.max(dfof_map[15:240,15:240]))
    vmin1 = (np.min(dfof_map[15:240,15:240]))
    #v=v/200
    fig = plt.figure(figsize=[9,6],constrained_layout=True)
    gs = fig.add_gridspec(nrows=8,ncols=8)
    ax1 = fig.add_subplot(gs[:9,:9])
    
    #plot with o in middle
    #norm = mcolors.DivergingNorm(vmin=-100, vmax = 100, vcenter=0)
    #plt.imshow(dfof_map,cmap='RdBu_r', norm=norm)
    
    plt.imshow(dfof_map,cmap='RdBu_r',vmin=-80, vmax = 80)
    #plt.imshow(im,cmap='RdBu_r',vmin=vmin1, vmax = vmax1)
    #plt.title('%dF/F Map')
    cbar = plt.colorbar()
    cbar.set_label('% dF/F', fontsize=20) #rotation=270

    turn_off_ticks = 0
    if turn_off_ticks:
        plt.xticks([])
        plt.yticks([])
    #pat = 'st'
    #plt.savefig(os.path.join(file_path,'dfof_map_RIOs'+infile_stim[:-3]+'.png'),bbox_inches='tight')

    roi_size = analysis_params['roi_size']
    #xy_rois_up #xy up left corner from xy_rois-half_size_of_ROI
    rect = []
    rois = []
    avg_rois = [] #average of fluorescence values in ROIs
    rois_dfof = []
    avg_rois_dfof = [] #average of dF/F values in ROIs

    for i in range(n_rois):
        if nr > 0: #n_pats=2: 2 pattern, 1st for stim, 2nd for retravel
            if i <= (n_rois-nr-1): 
                rectm = mpl.patches.Rectangle((xy_rois_up[i,0],xy_rois_up[i,1]),roi_size,roi_size,linewidth=2,edgecolor='b',facecolor='none')
            else:
                rectm = mpl.patches.Rectangle((xy_rois_up[i,0],xy_rois_up[i,1]),roi_size,roi_size,linewidth=2,edgecolor='r',facecolor='none')          
        elif nr == 0:                
            rectm = mpl.patches.Rectangle((xy_rois_up[i,0],xy_rois_up[i,1]),roi_size,roi_size,linewidth=2,edgecolor='k',facecolor='none')
        rect.append(rectm)
        #print(np.shape(rect))
        ax1.add_patch(rect[i])
        ax1.annotate(i+1, xy=(xy_rois_up[i,0],xy_rois_up[i,1]), xytext=(xy_rois_up[i,0],xy_rois_up[i,1]-2))
        
        roism = im_s[:,xy_rois_up[i,1]:xy_rois_up[i,1]+roi_size, xy_rois_up[i,0]:xy_rois_up[i,0]+roi_size]
        rois.append(roism)
        avg_roim = np.mean(roism,axis=(1,2))
        avg_rois.append(avg_roim)
        #print('ave_rios: ', np.shape(avg_rois))
        
        roism_dfof = dfof[:,xy_rois_up[i,1]:xy_rois_up[i,1]+roi_size, xy_rois_up[i,0]:xy_rois_up[i,0]+roi_size]
        rois_dfof.append(roism_dfof)
        avg_roim_dfof = np.mean(roism_dfof,axis=(1,2))
        avg_rois_dfof.append(avg_roim_dfof)
        
    
    ft = trial_params['ft1']
    x_time = ft*np.arange(0,np.shape(avg_rois)[1]) #time in second
                                      
    #pat = 'st'
    if savefig_name:
        plt.savefig(os.path.join(file_path,'000'+savefig_name+'_dfof_rois.png'),bbox_inches='tight')
        print('Figure saved at ',os.path.join(file_path,'000'+savefig_name+'dfof_map_RIOs'+'.png'))
                        
    avg_rois = np.array(avg_rois)
    print('avg_rois: ', np.shape(avg_rois))
    avg_rois_dfof = np.array(avg_rois_dfof)
    print('avg_rois_dfof: ', np.shape(avg_rois_dfof))
    
    f2,ax2 = plt.subplots(figsize=(10,5))
    plt.plot(x_time, avg_rois.T)

    #set stim marker    
    ax2.set_ylabel('Fluorescence', fontsize=20)
    #plt.axvspan(59, 60, color='red', alpha=0.25,label='Stim On')
    plt.axvspan(1*nPreFr*ft, 1*(nPreFr+nStimFrs_cut)*ft, color='red', alpha=0.25,label='Stim On') # *0.03 for time in second
    #plt.xlim([50,100])    
    leg = ax2.get_legend()
    #leg.legendHandles[0].set_color((0,0,1))
    #leg.legendHandles[1].set_color((1,0,0))

    
    #plot dFoF of traces
    f3,ax3 = plt.subplots(figsize=(10,5))
    #plot dF
    #plt.plot(x_time, avg_rois[:nr,:].T, color='red') # not retrival cells
    #plt.plot(x_time, avg_rois[nr:,:].T, color='blue') # retrival cells
    #plot dF/F
    #plt.plot(x_time, avg_rois_dfof[:(n_rois-nr),:].T, color=(0,0,1)) # n_rois: number of ROIs; not retrival cells
    plt.plot(x_time, avg_rois_dfof.T, color='red') # retrival cells
    #plt.plot(x_time, avg_rois_dfof[(n_rois-nr):(n_rois+1),:].T, color='red') # retrival cells
    plt.plot(x_time, np.mean(avg_rois_dfof[:(n_rois-nr),:], axis=0), color='mediumslateblue', linewidth=6) # n_rois: number of ROIs; not retrival cells
    #plt.plot(x_time, np.mean(avg_rois_dfof[(n_rois-nr):(n_rois+1),:], axis=0), color='tomato', linewidth=6) # retrival cells


    #plt.plot(x_time, avg_rois_dfof[n_rois,:].T, color='green')
    print('nr: ', nr)
    print('n_rois: ', n_rois)
    
#make one trace cyan
    #plt.plot(x_time, avg_rois_dfof[0,:].T, color='cyan')
    #plt.legend(['not retrival', 'retrival cells', 'other'])
    #plt.legend(['retrival cells','not retrival',  'other'])
    #ax = plt.gca.color('red')
    #leg.legendHandles[2].set_color('green')
#change stim marker    
    plt.axvspan(1*nPreFr*ft, 1*(nPreFr+nStimFrs_cut)*ft, color='red', alpha=0.25,label='Stim On') # *0.03 for time in second
    plt.xlim([1*(nPreFr-30)*ft,1*(nPreFr+50)*ft])
    #plt.gca().set_ylim(top=150)
    #plt.gca().set_ylim(bottom=-50)
    ax3.set_ylabel('dF/F (%)', fontsize=20)
    #plt.savefig(file_path+'test'+'_dfof_rois_traces.svg', format = 'svg', dpi=300)
    if savefig_name:
        plt.savefig(os.path.join(file_path,'000'+savefig_name+'_dfof_rois_traces.png'),bbox_inches='tight')
        print('Figure saved at ',os.path.join(file_path,'000'+savefig_name+'dfof_map_RIOs_traces'+'.png'))
        #plt.savefig(os.path.join(file_path,fname[:-4]+'_dfof_rois_traces.svg'), format = 'svg', dpi=300)
    
    return avg_rois_dfof, avg_rois #, f0_rois, F0_min
    #return avg_rois, xy_rois

##----------------------------------------------------------------
    
def generate_dfof_cell_tracesAvg(raw_traces,im,trial_params,nCell):
    '''apply cell masks to an image and calculate the average fluorescence, then 
    dF and dF/F values across all frames'''
    # set dF/F parameters
    print('im: [frame,x,y] for average images over trails')
    nTrial = trial_params['nTrial']
    nFrTrial = trial_params['nFrTrial']
    nPreFr = trial_params['nPreFr']
    #nCell = trial_params['nCell']
    br = trial_params['baseline_range']
    
    print(nFrTrial,im.shape[1],im.shape[2])

    # find background subtraction value from across pre-stim frames (minimum pixel value)
    im_raw = im.reshape((nFrTrial,im.shape[1],im.shape[2]))

#     # subtract background value from all traces
    Fraw_min = np.mean(im_raw, axis=0).min()
    F = raw_traces - Fraw_min
    F[F<0]=0

    # calculate baseline value for each cell (avg across all pre-stim frames from all trials; resulting shape: (nCell))
    baseF0 = F.reshape((nCell,nTrial,nFrTrial))
    baseF = baseF0[:,:,br[0]:br[1]].mean(axis=(1,2))
    print('Updated df/f Calculation')
    
    # calculate dF and dF/F value for every cell at every frame; shape (nCell,nTrial,nFrTrial)
    dF = np.zeros(F.shape)
    dFoF = np.zeros(F.shape) #(1240, 12000)    
    
    for iC in range (nCell):
        for iF in range (nTrial*nFrTrial):
            dF[iC,iF] = F[iC,iF]-baseF[iC]
            dFoF[iC,iF] = (dF[iC,iF]/baseF[iC]) * 100

    return dFoF,dF,F,baseF    




#-----------------------------------------------------------
#stim artifact removal
def artif_remov_suite2p(dFoF_trialavg, trial_params):
    print('to remove stimulus artifact for subtracting the noise (1st stim frame - ave(base frames)')
    nPreFr = trial_params['nPreFr']
    baseline_range = trial_params['baseline_range']
    nStimFrs_cut = trial_params['nStimFrs_cut']

    noise = dFoF_trialavg[:,nPreFr] - np.mean(dFoF_trialavg[:,baseline_range], axis=1)
    print('noise.shape: ', noise.shape)

    stimFrs_corrected = []
    for i in range(nStimFrs_cut):
        mid0 = dFoF_trialavg[:,nPreFr+i] - noise
        stimFrs_corrected.append(mid0)  
    stimFrs_corrected = np.transpose(np.array(stimFrs_corrected))
    print('stimFrs_corrected.shape: ', stimFrs_corrected.shape)

    print('dFoF_trialavg[:,:nPreFr].shape: ', dFoF_trialavg[:,:nPreFr].shape)
    print('dFoF_trialavg[:,nPreFr+nStimFrs_cut:].shape: ', dFoF_trialavg[:,nPreFr+nStimFrs_cut:].shape)

    dFoF_trialavg_corred = np.concatenate((dFoF_trialavg[:,:nPreFr], stimFrs_corrected, dFoF_trialavg[:,nPreFr+nStimFrs_cut:]), axis=1)
    print('dFoF_trialavg_corred.shape: ', dFoF_trialavg_corred.shape)
    return dFoF_trialavg_corred



#-------------------------------------------------
#add ROI plot in image

#global file_path2
def ROI_in_image(im_frame,xy_rois,roi_size,file_path2save):
    #nr2 = nr2
    #im_s1 = tfl.imread(file_path+'/'+im_file)
    fig = plt.figure(figsize=[6.922,6.922],frameon=False) #6.922, constrained_layout=True, 
    #gs = fig.add_gridspec(nrows=8,ncols=8)
    #ax1 = fig.add_subplot(gs[:9,:9])
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax1.axis('off')
    
    #xy_rois = genfromtxt(file_path + '/'+ xy_file, delimiter=',')
    #np.delete(xy_rois[:],0,axis=0)
    
    xy_rois = (xy_rois*2 - roi_size/2).astype(int) #xy_rois*2 --> 256x256 to 512x512; -roi_size/2 --> upleft coner of the roi squares
    #xy_rois = (xy_rois[1:,1:3]*um2pix).astype(int) #xy coordinates from original auto saved SLM pattern generator
    #xy_rois = xy_rois[1:,1:3].astype(int) #xy coordinates from 256x256 pixels' image
    #xy_rois = xy_rois*2 # set [256,256] back to [512, 512]

    #print('xy_rois:', xy_rois)
    
#rectengular rois
    #xy_rois = xy_rois-int(roi_size/2) #convert the center to up left
    #display('xy_rois: ', xy_rois) #col0: value; col1: x; col2: y
    #print(xy_rois[2,3])
    #print('xy_rois:', xy_rois)
    n_rois = np.shape(xy_rois)[0]
    #nr = 12 #nr: number of cells for retrival
    print('xy_rois: ', np.shape(xy_rois))
    #print('xy_rois[1,:]: ', xy_rois[1,:])
    
    nr = int(input('nr: pattern B cell number'))
    nr2 = int(input('nr2: pattern C cell number'))
    
    image0 = plt.imshow(im_frame)#, cmap='gray', interpolation='none')#, vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    
    rect = []
    rois = []
    avg_rois = [] #average of values in ROIs

# dx=0, dy=0 for rectangular    
    dx = 0 #-20,manually adjust x, y coordinates -20
    dy = 0 #-5
    for i in range(n_rois): #n_rois,  use 0 to not to have ROIS
        if i < n_rois-nr-nr2: #n_pats=2: 2 pattern, 1st for stim, 2nd for retravel
            rectm = mpl.patches.Rectangle((xy_rois[i,0]+dx,xy_rois[i,1]+dy),roi_size,roi_size,linewidth=1,edgecolor='b',facecolor='none')
            #rectm = mpl.patches.Circle((xy_rois[i,0],xy_rois[i,1]),roi_size,linewidth=1,edgecolor='b',facecolor='none')    
            ax1.annotate(i+1, xy=(xy_rois[i,0]+dx,xy_rois[i,1]-dy), xytext=(xy_rois[i,0]+dx,xy_rois[i,1]+dy-2), color='b')
        elif (i >= n_rois-nr-nr2) & (i < n_rois-nr2):
            rectm = mpl.patches.Rectangle((xy_rois[i,0]+dx,xy_rois[i,1]+dy),roi_size,roi_size,linewidth=1,edgecolor='magenta',facecolor='none')          
            #circles
            #rectm = mpl.patches.Circle((xy_rois[i,0],xy_rois[i,1]),roi_size,linewidth=1,edgecolor='magenta',facecolor='none')                 
            ax1.annotate(i+1, xy=(xy_rois[i,0]+dx,xy_rois[i,1]-dy), xytext=(xy_rois[i,0]+dx,xy_rois[i,1]+dy-2), color='magenta')
        elif (i > n_rois-nr2):
            rectm = mpl.patches.Rectangle((xy_rois[i,0]+dx,xy_rois[i,1]+dy),roi_size,roi_size,linewidth=1,edgecolor='w',facecolor='none')          
            #circles
            #rectm = mpl.patches.Circle((xy_rois[i,0],xy_rois[i,1]),roi_size,linewidth=1,edgecolor='k',facecolor='none')                 
            ax1.annotate(i+1, xy=(xy_rois[i,0]+dx,xy_rois[i,1]-dy), xytext=(xy_rois[i,0]+dx,xy_rois[i,1]+dy-2), color='w')

        rect.append(rectm)
        #print(np.shape(rect))
        ax1.add_patch(rect[i])
        #rectangular
        #ax1.annotate(i+1, xy=(xy_rois[i,0]+dx,xy_rois[i,1]+dy), xytext=(xy_rois[i,0]+dx,xy_rois[i,1]+dy-2))
        #circle
        #ax1.annotate(i+1, xy=(xy_rois[i,0]+dx,xy_rois[i,1]-dy), xytext=(xy_rois[i,0]+dx,xy_rois[i,1]+dy-2), color='w')
        
        #roism = im_s1[:,xy_rois[i,1]+dy:xy_rois[i,1]+dy+roi_size, xy_rois[i,0]+dx:xy_rois[i,0]+dx+roi_size]
        #rois.append(roism)
        #avg_roim = np.mean(roism,axis=(1,2))
        #avg_rois.append(avg_roim)
        #print('ave_rios: ', np.shape(avg_rois))
    
    #add scale bar
    #zoom x2, 512x512um, 50um->50pixels; zoom x2.5, 512x512 pixels -> 410x410um, 50um->62.4 pixels
    scaleBar= mpl.patches.Rectangle((400,495), 50, 4, color='w') #zoom x3, 512x512 pixels -> 314x314um, 50um->81.5 pixels
    ax1.add_patch(scaleBar)
    ax1.text(402,489,'50 \N{greek small letter mu}m', fontsize=15, color='w')#00B5 micro, \N{00B5}
        
    #global file_path2
    # %store -r file_path2p 
    print('print file_path2p inside function: ', file_path2)
    plt.savefig(os.path.join(file_path2save,'image_rois_scale.png'),bbox_inches='tight') #fname[:-4]
    #plt.savefig(os.path.join(file_path,im_file[:-4]+'_image_rois.tif'),bbox_inches='tight') #fname[:-4]
    
    return image0, fig
    
    
