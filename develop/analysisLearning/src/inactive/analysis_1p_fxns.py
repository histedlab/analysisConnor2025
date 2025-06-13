import numpy as np
import pandas as pd
import scipy.io
import os,sys
import math
import tifffile
from pathlib import Path
from skimage import transform

import mworksbehavior as mwb
from mworksbehavior import mwk_io
from mworksbehavior import mwkfiles
from mworksbehavior import mat_io

import pytoolsMH as ptMH

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter

import seaborn as sns
sns.set_style('ticks')
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(category=FutureWarning, action="ignore")
    import caiman as cm

from IPython.display import clear_output

r_ = np.r_
a_ = np.asarray


def plot_trace(y, pres, post, trial_start):
    ''' Plots the cumulative stack of tifs and marks the trials and the indicies
    of the baseline frames and post-stimulus response frames
    
    Arguments:
        y = the mean df/f data per frame
        pres = baseline start frames
        post = indicies for post-stimulus frames
        trial_start = start frame for the trial
        
    Returns:
        One plot with the appropriate markings'''
    fig = plt.figure(1, figsize=(15, 5))
    x = np.linspace(0,len(y), num=len(y))
    lines = post
    pre = pres
    plt.plot(x,y, c='black', linewidth=0.5)
    for l in lines:
        plt.axvline(l, c='r', ls='--')
    for p in pre:
        plt.axvline(p, c='b', ls='--')
    for tr in trial_start:
        plt.axvline(tr, c='g')
    plt.show()


def make_trial_figure(datadir, df, n, rows, cols):
    '''Plots a subset of trials to check for alignment. 
    Arguments: 
        datadir = the path to where the data is located
        df = the post-aligned dataframe containing the tif files and stim
        n = number of trials you want to plot
        rows = number of rows you want
        cols = number of columns you want 
    
    Returns:
        Plots of individual trials to show alignment of the stimulus and response. '''
    rows = rows
    cols = cols
    size = (15, 8)
    fig = plt.figure(1, figsize=size)
#     plt.suptitle(f'Avg dF/F for Decrement Response', fontsize=20)
    for i in range(n):
        fig.add_subplot(rows, cols, i+1)
        plt.title(f'Trial {df.tifs[i]} - {df.tLaserPowerMw[i]} mW', fontsize=15)
        im = plot_trial(datadir, df, i)
    plt.tight_layout()
    
    plt.show()


def plot_trial(datadir, df, i):
    ''' The plotting function that is used in make_trial_figure. 
    Arguments:
        datadir = the path to the data folder
        df = the dataframe containing the tif file names and stimulation frame
        i = trial number 
    
    Returns:
        A single plot that is used in make_trial_figure'''
    trial = tifffile.imread(os.path.join(datadir, df.tifNames.iloc[i]))
    plt.axvline(df.stim_fr.iloc[i], c='g')
    y = []
    for i in range(trial.shape[0]):
        y.append(trial[i].mean())
    x = np.linspace(0,len(y), num=len(y))
    plt.plot(x,y, c='black', linewidth=0.5)


def sort_powers(tr_info):
    '''Sorts the trials for each power level to run the main figure function smoothly. 
    Checks to make sure the trial has at least 10 frames before the stimulus so there is a sufficient baseline
    
    Arguments: 
        tr_info = the dataframe that contains the trial info
        powers = the original list of powers
        
    Returns:
        A sorted list of approved powers. '''
    powers = tr_info.tLaserPowerMw.unique()[::-1]
    new_powers = []
    for power in powers:
        tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
        sub_info = tr_info.iloc[tifNums]

        base_shift = 10 #base frames set at 50 frames before stimulation occurs
        res_shift = 0 #response frames set at 0 frames after the stimulation occurs (1P has blanking)

        stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
        base_fr = (stim_fr - base_shift).tolist()
        res_fr = (stim_fr + res_shift).tolist()
        stim_fr = stim_fr.tolist()
        count = 0
        for file_index in range(len(base_fr)):
            if (base_fr[file_index] > 0):
                count +=1
        if count > 0:
            new_powers.append(power)
    return sorted(new_powers)


def plot_power_grid(datadir, date, animal, tr_info, power_val, blanking, smoothed, sigma):
    ''' Plots the df/f graphs at each power after motion correction
    Arguments:
        datadir = the path to the data folder
        date = the date the data was collected
        animal = the animal used in the experiment
        tr_info = the data frame with trial data
        power_val = the list of powers that will produce graphs
        
    Returns: 
        im = an image that is ready to be plotted 
    '''
    
    power = power_val
    tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
    tifNames = tr_info.tifNames[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)]


    # load files and stack into one big file
    # remove frames with 1p artifact
    ims = []

    totalfr = 0
    for i, tif in enumerate(tifNames):
        print(i, tif)
        trIn = os.path.join(datadir, tif)
        trOut = os.path.join(datadir, 'py_{}'.format(tif))

        im = SI_batch_resave(trIn, trOut, nFrChunk=1000, downscaleTuple=(1,2,2), rewriteOk=True)
        print(im)

        ims.append(im)
        clear_output(wait=True)

    ims = np.asarray(ims)

    stack = np.vstack(ims)
    shapes = [im.shape[0] for im in ims]   

    np.save(os.path.join(
         datadir, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)), shapes)

    tifffile.imsave(os.path.join(
         datadir, 'i{}-{}-{}-stack.tif'.format(animal, date, power)), stack, bigtiff=True)
    
    #Motion Correction

    infiles = os.path.join(datadir,'i{}-{}-{}-stack.tif'.format(animal, date, power))
    infiles = [infiles]

    # define downscaling factors
    scale = (1,2,2) # (1,2,2) (t,x,y)

    opts_mc = {
    # dataset dependent parameters
    'fnames': infiles,             
    'fr': 30,                     # imaging rate in frames per second
    # motion correction parameters  
    'strides': (48, 48),          # start a new patch for pw-rigid motion correction every x pixels
    'overlaps': (24, 24),         # overlap between pathes (size of patch strides+overlaps)
    'max_shifts': (6,6),          # maximum allowed rigid shifts (in pixels)
    'max_deviation_rigid': 3,     # maximum shifts deviation allowed for patch with respect to rigid shifts
    'pw_rigid': True,             # flag for performing non-rigid motion correction
    }

    opts = cm.source_extraction.cnmf.params.CNMFParams(params_dict=opts_mc)

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=1, single_thread=True)

    # first we create a motion correction object with the parameters specified
    mc = cm.motion_correction.MotionCorrect(infiles, dview=dview, **opts.get_group('motion'))

    #%% Run piecewise-rigid (or not) motion correction using NoRMCorre
    mc.motion_correct(save_movie=True)
    if opts.motion['pw_rigid']:
        m_els = cm.load(mc.fname_tot_els)  # for pw_rigid
    else:
        m_els = cm.load(mc.fname_tot_rig)  # for rigid
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 
        # maximum shift to be used for trimming against NaNs

    if opts.motion['pw_rigid']:
        newname = os.path.join(datadir,f'renorm-mc-{power}.tif')
    else:
        newname = os.path.join(datadir,f'renorm-mc-rigid-{power}.tif')
    tifffile.imsave(newname, m_els)


    #Generate dF/F
    sub_info = tr_info.iloc[tifNums]

    base_shift = 10 #base frames set at 50 frames before stimulation occurs
    if blanking == True:
        res_shift = 0 #response frames set at 0 frames after the stimulation occurs (1P has blanking)
    if blanking == False:
        res_shift = 8
    stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
    base_fr = (stim_fr - base_shift).tolist()
    #stim_fr = (tr_info.trHoldFrs - tr_info.trReactionTimeFrs).tolist()
    res_fr = (stim_fr + res_shift).tolist()
    stim_fr = stim_fr.tolist()

    #load stacked movies
    im_stim = tifffile.imread(os.path.join(datadir, f'renorm-mc-{power}.tif'))

    #df/f avg all trials
    past_fr = 0
    baseline = im_stim.min()
    sumFo = 0
    sumF = 0
    avg_pre = 10
    avg_post = 10
    tr_count = 0

    #for file_index in range(len(shapes)):
    for file_index in range(len(base_fr)):
        if (base_fr[file_index] > 0):

            tr_count += 1

            #print(im_stim[int(past_fr + base_fr[0]), int(past_fr + base_fr[0] + avg_len)])
            sumFo += im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline

            #print(past_fr + int(base_fr[file_index]), past_fr + int(base_fr[file_index]) + avg_len)
            sumF += im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline

        past_fr += shapes[file_index]


    sumFo = sumFo / tr_count
    sumF = sumF / tr_count
    if smoothed==False:
        sumdFF = (sumF - sumFo) / sumFo 
    elif smoothed==True:
        sumdFF = (sumF - sumFo) / gaussian_filter(sumFo, sigma=20)
    dFF = sumdFF * 100    

    im = plt.imshow(dFF, cmap = "RdBu_r")
    plt.clim(-60,60)
    return im, tr_count

def plot_power_grid_no_mc(datadir, date, animal, tr_info, power_val, blanking, smoothed, sigma): 
    ''' Creates a data frame with the average df/f value for each trial at a given ROI
    
    Arguments:
        date: string of the date the data was collected
        animal: string of the number for the animal 
        success_only: True or False if you want the dataframe to only include success trials
        set_roi: True or False if you want a specific ROI selected 
        x, y, width, height: the same coordinates used in find_roi() will go here to retrieve data for that specific ROI
        
    Returns:
        tr_info: data frame with all of the average df/f values appened for each trial, either in total or only success trials
    '''
#     datadir_ = f'{datadir}{date}-i{animal}'
    tr_info_ = tr_info
    
    power = power_val
    

    tifNums = tr_info_[(tr_info.trOutcome == 'success') & (tr_info_.tLaserPowerMw == power)].index.tolist()
    tifNames = tr_info_.tifNames[(tr_info.trOutcome == 'success') & (tr_info_.tLaserPowerMw == power)]

    
    shapes = np.load(os.path.join(datadir, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)))


    # Get Df/f info
    sub_info = tr_info.iloc[tifNums].reset_index()

    base_shift = 10 #base frames set at 50 frames before stimulation occurs
    if blanking == True:
        res_shift = 0 #response frames set at 0 frames after the stimulation occurs (1P has blanking)
    if blanking == False:
        res_shift = 8
    stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
    base_fr = (stim_fr - base_shift).tolist()
    #stim_fr = (tr_info.trHoldFrs - tr_info.trReactionTimeFrs).tolist()
    res_fr = (stim_fr + res_shift).tolist()
    stim_fr = stim_fr.tolist()
    im_stim = tifffile.imread(os.path.join(datadir, f'renorm-mc-{power}.tif'))

    #df/f avg all trials
    past_fr = 0
    baseline = im_stim.min()

    sumFo = 0
    sumF = 0
    avg_pre = 10
    avg_post = 10
    tr_count = 0
    sumFos = []
    sumFs = []
    
    for file_index in range(len(base_fr)):

        if (base_fr[file_index] > 0):

            tr_count += 1

            #print(im_stim[int(past_fr + base_fr[0]), int(past_fr + base_fr[0] + avg_len)])
            sumFo += im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline

            #print(past_fr + int(base_fr[file_index]), past_fr + int(base_fr[file_index]) + avg_len)
            sumF += im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline

        past_fr += shapes[file_index]

    sumFo = sumFo / tr_count
    sumF = sumF / tr_count
    if smoothed==False:
        sumdFF = (sumF - sumFo) / sumFo 
    elif smoothed==True:
        sumdFF = (sumF - sumFo) / gaussian_filter(sumFo, sigma=sigma)
    dFF = sumdFF * 100    

    im = plt.imshow(dFF, cmap = "RdBu_r")
    plt.clim(-60,60)
    return im, tr_count


def make_figure(datadir, date, animal, tr_info, new_powers, blanking, mc=True, scale=1, smoothed=True, sigma=0):
    ''' Creates the combined figure of df/f plots for each power on a given day. 
    Arguments: 
        datadir = the path to the data folder
        date = the date the data was collected 
        animal = the animal used in the experiment 
        tr_info = the data frame with the trial data 
        new_powers = the list of sorted powers 
        scale = if power has been attenuated, the scaling factor is set here. Otherwise it is set to 1.
        blanking = True if there is blanking (1p), False if there is no blanking (2p).
        mc = True if you want motion correction, False if you have run motion correction already and want to skip it
        smoothed = True if you want a smoothed denominator for the df/f plot, False if no smoothing is wanted
        sigma = the gaussian filter smoothing value'''
    
    title_powers = np.round(np.array(new_powers)*scale, 3)
    if len(new_powers)%2==0:
        rows = 2
        cols = len(new_powers)/2
        size = (17, 7)
    elif len(new_powers)%3==0:
        rows = 3
        cols = len(new_powers)/3
        size = (15, 15)
    else:
        rows = 3
        cols = 4
        size = (15, 8)
    fig = plt.figure(1, figsize=size)
#     plt.suptitle(f'Avg dF/F for Decrement Response', fontsize=20)
    for i in range(len(new_powers)):
        fig.add_subplot(rows, cols, i+1)
        if mc==True:
            im, n = plot_power_grid(datadir, date, animal, tr_info, new_powers[i], blanking, smoothed, sigma)
        if mc==False:
            im, n = plot_power_grid_no_mc(datadir, date, animal, tr_info, new_powers[i], blanking, smoothed, sigma)
        plt.title(f'{title_powers[i]}mW, n={n}', fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.65])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"{date}-i{animal}.png")
    
    plt.show()


def SI_batch_resave(infile, outfile, nFrChunk=300, rewriteOk=False, downscaleTuple=None):
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
        T = len(tif.pages)
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

    im = im.astype('int16')
    tifffile.imsave(outfile, im, bigtiff=True)
    print('done. saved to {}'.format(outfile))
    return im

def makePositive(im):
    if np.min(im) < 0:
        im += abs(np.min(im))
    return im

def create_trial_df(datadir, date, animal, success_only, set_roi=False, x=0, y=0, width=0, height=0, blanking=True):
    ''' Creates a data frame with the average df/f value for each trial at a given ROI
    
    Arguments:
        date: string of the date the data was collected
        animal: string of the number for the animal 
        success_only: True or False if you want the dataframe to only include success trials
        set_roi: True or False if you want a specific ROI selected 
        x, y, width, height: the same coordinates used in find_roi() will go here to retrieve data for that specific ROI
        
    Returns:
        tr_info: data frame with all of the average df/f values appened for each trial, either in total or only success trials
    '''
    datadir_ = f'{datadir}{date}-i{animal}'
    tr_info = pd.read_csv(os.path.join(datadir_, f'tr_info-{date}-{animal}.csv'))
    tr_info = tr_info.drop('Unnamed: 0', axis=1)
    tr_info['dffs'] = np.zeros(tr_info.shape[0])
    
    new_powers = sort_powers(tr_info)
    
    for power in new_powers:

        tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
        tifNames = tr_info.tifNames[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)]

        shapes = np.load(os.path.join(datadir_, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)))


        # Get Df/f info
        sub_info = tr_info.iloc[tifNums].reset_index()

        base_shift = 10 #base frames set at 50 frames before stimulation occurs
        if blanking == True:
            res_shift = 0 #response frames set at 0 frames after the stimulation occurs (1P has blanking)
        if blanking == False:
            res_shift = 8
        stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
        base_fr = (stim_fr - base_shift).tolist()
        #stim_fr = (tr_info.trHoldFrs - tr_info.trReactionTimeFrs).tolist()
        res_fr = (stim_fr + res_shift).tolist()
        stim_fr = stim_fr.tolist()
        im_stim = tifffile.imread(os.path.join(datadir_, f'renorm-mc-{power}.tif'))

        #df/f avg all trials
        past_fr = 0
        baseline = im_stim.min()

        sumFo = 0
        sumF = 0
        avg_pre = 10
        avg_post = 10
        tr_count = 0
        sumFos = []
        sumFs = []
        for file_index in range(len(base_fr)):
            if (base_fr[file_index] > 0):

                tr_count += 1

        #         print(im_stim[int(past_fr + base_fr[0]), int(past_fr + base_fr[0] + avg_len)])           
                sumFos.append(im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline)


        #         print(past_fr + int(base_fr[file_index]), past_fr + int(base_fr[file_index]) + avg_len)
                sumFs.append(im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline)

            past_fr += shapes[file_index]


        sumdFFs = ((np.array(sumFs)-np.array(sumFos))/np.array(sumFos))*100
        avgs = []
        for dff in sumdFFs:
            if set_roi==True:
                avgs.append(dff[y:y+height, x:x+width].mean())
            elif set_roi==False:
                avgs.append(dff.mean())
        for i, idx in enumerate(tifNums):
            if set_roi==True:
                tr_info.dffs[idx] = sumdFFs[i][y:y+height, x:x+width].mean()
            elif set_roi==False:
                tr_info.dffs[idx] = sumdFFs[i].mean()
        tr_info.to_csv(os.path.join(datadir_, f'tr_info_dff_{date}.csv'))
    if success_only == True:
        return tr_info[tr_info.trOutcome == 'success']
    else:
        return tr_info


def plot_trial_data(success_df, date, scale=1):
    '''Plots each trial point for a given date
    
    Arguments:
        success_df: the success only trial data frame created with create_trial_df()
        date: string date of the day data was collected
        scale: if power was attenuated, the scale of attenuation goes here
        
    Returns:
        plot of the trial data and will save a figure under "color_coded_trial_{date}.png" '''
    fig = plt.figure(1, figsize=(15, 8))
    plt.scatter((success_df.tLaserPowerMw)*scale, success_df.dffs, c = success_df.index)
    plt.colorbar(label='Trial Number')
    plt.title(f'Trial data for {date}', fontsize=20)
    plt.xlabel('Power lever (mW)', fontsize=15)
    plt.ylabel('Average dF/F', fontsize=15)
    plt.savefig(f"color_coded_trial_{date}")

def plot_within_day_splits(success_df, size, lowess_frac=0.5, scale=1, image_name='within_day_splits.png'):
    ''' Plots the df/f averages for each power within a given day, split into halves or quarters
    
    Arguments:
        success_df: same data frame created with create_trial_df, success trials only
        size: "half" or "quarter" indicating how the data will be split
        lowess_frac: the amount of data used for smoothing, 0 to 1 with 1 being completely smoothed and 0 being no smoothing
        scale: how much the power was attenuated by if that is applicable
        
    Returns:
        A plot of the df/f averages by power split in half or in quarters. Also saves a figure under "{date}_{size}_lowess{lowess_frac}.png"'''
    date = success_df.trDate.iloc[0]
    if size == 'half':
        groups = [success_df[0:int(success_df.shape[0]/2)], success_df[int(success_df.shape[0]/2):]]
    elif size == 'quarter':
        idx = int(success_df.shape[0]/4)
        groups = [success_df[0:idx], success_df[idx:idx*2], success_df[idx*2:idx*3], success_df[idx*3:]]
    fig = plt.figure(1)
    all_vals = []
    for group in groups:
        dff_means = []
        dff_sems = []
        power, count = np.unique(group.tLaserPowerMw, return_counts=True)
        for power, count in zip(power, count):

            dff_mean = group.dffs[group.tLaserPowerMw==power].mean()
            dff_sem = group.dffs[group.tLaserPowerMw==power].std()/np.sqrt(count)
            dff_means.append(dff_mean)
            dff_sems.append(dff_sem)
        all_vals.append(dff_means)
        smooth_dffs = lowess(dff_means, np.unique(group.tLaserPowerMw), frac=lowess_frac, return_sorted=False)
        plt.plot((np.unique(group.tLaserPowerMw))*scale, smooth_dffs)
#         plt.plot(np.unique(group.tLaserPowerMw), dff_means)
        plt.fill_between(np.unique(group.tLaserPowerMw)*scale, y1=np.array(smooth_dffs)+np.array(dff_sems), y2=np.array(smooth_dffs)-np.array(dff_sems), alpha=0.25)
    if size == 'half':
        plt.legend(labels = ['First Half', 'Second Half'])
        plt.title(f'Half Split for {date}')
    elif size == 'quarter':
        plt.legend(labels = ['First Quarter', 'Second Quarter', 'Third Quarter', 'Fourth Quarter'])
        plt.title(f'Quarter Split for {date}')
    plt.ylabel('Average dF/F')
    plt.xlabel('Power Level (mW)')
    plt.savefig(f'{date}_{size}_lowess{lowess_frac}_{image_name}.png')
    return all_vals

def find_roi(datadir, date, animal, x, y, width, height, blanking=True):
    '''A visual for finding the desired region of interest
    
    Arguments:
        date: string of the date the data was collected
        animal: string of the animal id number 
        x: the column value for the left side of the roi
        y: the row value for the top of the roi 
        width: the number of columns to span across
        height: the number of rows to span across 
    
    Returns:
        An image of the date's df/f plot with a roi outlined in red'''
    
    datadir_ = f'{datadir}{date}-i{animal}'
    tr_info = pd.read_csv(os.path.join(datadir_, f'tr_info-{date}-{animal}.csv'))
    tr_info = tr_info.drop('Unnamed: 0', axis=1)
    new_powers = sort_powers(tr_info)

    power = new_powers[-2]

    tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
    tifNames = tr_info.tifNames[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)]


    shapes = np.load(os.path.join(datadir_, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)))


    # Get Df/f info
    dFF = 0
    sub_info = tr_info.iloc[tifNums].reset_index()

    base_shift = 10 
    if blanking == True:
        res_shift = 0 #response frames set at 0 frames after the stimulation occurs (1P has blanking)
    if blanking == False:
        res_shift = 8

    stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
    base_fr = (stim_fr - base_shift).tolist()

    res_fr = (stim_fr + res_shift).tolist()
    stim_fr = stim_fr.tolist()

    im_stim = tifffile.imread(os.path.join(datadir_, f'renorm-mc-{power}.tif'))

    #df/f avg all trials
    past_fr = 0
    baseline = im_stim.min()

    sumFo = 0
    sumF = 0
    avg_pre = 10
    avg_post = 10
    tr_count = 0

    for file_index in range(len(base_fr)):
        if (base_fr[file_index] > 0):

            tr_count += 1

            sumFo += im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline

            sumF += im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline

        past_fr += shapes[file_index]


    sumFo = sumFo / tr_count
    sumF = sumF / tr_count
    sumdFF = (sumF - sumFo) / gaussian_filter(sumFo, sigma=20)

    dFF += sumdFF *100

    im = dFF
    
    # Plot
    fig, ax = plt.subplots()
    plt.imshow(im, cmap = "RdBu_r")
    plt.clim(-60,60)
    plt.colorbar(label='d F/F')
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    plt.show()

def compare_days_dff(datadir, dates, animal, set_roi, x, y, width, height, image_name, blanking=True):
    '''Plots each day's df/f curves for a direct comparison
    
    Arguments: 
        dates: a list of all of the dates with each date in string format
        animal: string of the animal id number 
        set_roi: True or False if you want a specfic region of the image 
        x, y, width, height: same coordinate values used in find_roi() to get data only from that ROI
        image_name: a string to name the image when you save it
    
    Returns:
        A plot of the day comparisons and saves the plot under the name given in image_name'''
    
    fig = plt.figure(1, figsize=(15, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, day in enumerate(dates):

        dFF_avgs = []
        sems = []
        date = day

        datadir_ = f'{datadir}{date}-i{animal}'
        tr_info = pd.read_csv(os.path.join(datadir_, f'tr_info-{date}-{animal}.csv'))
        tr_info = tr_info.drop('Unnamed: 0', axis=1)
        new_powers = sort_powers(tr_info)

        for val in new_powers:
            power = val

            tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
            tifNames = tr_info.tifNames[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)]


            shapes = np.load(os.path.join(datadir_, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)))


            # Get Df/f info
            sub_info = tr_info.iloc[tifNums].reset_index()

            base_shift = 10 
            if blanking == True:
                res_shift = 0 
            if blanking == False:
                res_shift = 8

            stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
            base_fr = (stim_fr - base_shift).tolist()

            res_fr = (stim_fr + res_shift).tolist()
            stim_fr = stim_fr.tolist()

            im_stim = tifffile.imread(os.path.join(datadir_, f'renorm-mc-{power}.tif'))

            #df/f avg all trials
            past_fr = 0
            baseline = im_stim.min()

            sumFo = 0
            sumF = 0
            avg_pre = 10
            avg_post = 10
            tr_count = 0
            sumFos = []
            sumFs = []
            for file_index in range(len(base_fr)):
                if (base_fr[file_index] > 0):

                    tr_count += 1

                    sumFo += im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline
                    sumFos.append(im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline)

                    sumF += im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline
                    sumFs.append(im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline)

                past_fr += shapes[file_index]

            sumdFFs = ((np.array(sumFs)-np.array(sumFos))/np.array(sumFos))*100
            avgs = []
            for dff in sumdFFs:
                if set_roi == True:
                    avgs.append(dff[y:y+height, x:x+width].mean())
                elif set_roi == False: 
                    avgs.append(dff.mean())

            dFF_avgs.append(np.array(avgs).mean())
            sems.append(np.array(avgs).std()/np.sqrt(tr_count))

        if day=='210511':
            new_powers = np.array(new_powers)*0.5
        if day=='210512':
            new_powers = np.array(new_powers)*0.5
        if day=='210513':
            new_powers = np.array(new_powers)*0.4
        log = np.log10(np.array(new_powers))
        plt.plot(new_powers, dFF_avgs, c=colors[i])
        plt.legend(labels=dates, prop={'size':11})
        plt.fill_between(new_powers, y1=np.array(dFF_avgs)+np.array(sems), y2=np.array(dFF_avgs)-np.array(sems), alpha=0.25, color=colors[i])
        plt.xlabel('Power level (mW)', fontsize=15)
        plt.ylabel('Average dF/F', fontsize=15)

    plt.savefig(f"{image_name}.png")


def slope_change(datadir, date, animal, lowess_frac):
    '''Looks at the change in slope within a given date
    
    Arguments:
        date: date the data was collected
        animal: id number of the animal 
        lowess_frac: how much you want to smooth the data by, 0-1 with 1 being completely and 0 being none
        
    Returns:
        3 plot figure with the df/f curve, the slope change, and the smoothed slope change'''
    
    datadir_ = f'{datadir}{date}-i{animal}'
    df = pd.read_csv(os.path.join(datadir_, f'tr_info_dff_{date}.csv'))
    powers = sort_powers(df)
    success = df[df.trOutcome=='success']
    ys=[]
    for power in powers:
        ys.append(success.dffs[success.tLaserPowerMw==power].mean())
    slopes = [((ys[0]-0)/(powers[0]-0))]
    for i in range(0, len(powers)-1):
        slope = ((ys[i+1]-ys[i])/(powers[i+1]-powers[i]))
        slopes.append(slope)
    fig = plt.figure(1, figsize=(20, 4))
    fig.add_subplot(1, 3, 1)
    plt.plot(powers, ys)
    plt.title('Average dF/F by Power Level')
    fig.add_subplot(1, 3, 2)
    plt.plot(powers, slopes)
    plt.title('Average Slope')
    fig.add_subplot(1, 3, 3)
    plt.plot(powers, lowess(slopes, powers, frac=lowess_frac, return_sorted=False))
    plt.title('Average Slope with Lowess Smoothing')
    plt.show()

def slope_change_within_day(datadir, date, animal, lowess_frac):
    '''Plots the changes in slope within a day split in half
    
    Arguments: 
        date: date the data was collected
        animal: animal id number 
        lowess_frac: the amount of smoothing with 0 being none and 1 being completely
        
    Returns:
        A plot with the slope change for the day split in half '''
    
    df = pd.read_csv(os.path.join(f'{datadir}{date}-i{animal}', f'tr_info_dff_{date}.csv'))

    success = df[df.trOutcome=='success']
    # if size == 'half':
    groups = [success[0:int(success.shape[0]/2)], success[int(success.shape[0]/2):]]
    # elif size == 'quarter':
    #     idx = int(success_df.shape[0]/4)
    #     groups = [success_df[0:idx], success_df[idx:idx*2], success_df[idx*2:idx*3], success_df[idx*3:]]
    fig = plt.figure(1, figsize=(10, 4))
    for group in groups:
        ys = []
        powers = np.unique(group.tLaserPowerMw)[::-1]
        for power in powers:
            dff_mean = group.dffs[group.tLaserPowerMw==power].mean()
            ys.append(dff_mean)

        slopes = [((ys[0]-0)/(powers[0]-0))]
        for i in range(0, len(powers)-1):
            slope = ((ys[i+1]-ys[i])/(powers[i+1]-powers[i]))
            slopes.append(slope)

        plt.plot(powers, lowess(slopes, powers, frac=lowess_frac, return_sorted=False))

    plt.legend(labels = ['First Half', 'Second Half'])
    plt.ylabel('Slope', fontsize=15)
    plt.xlabel('Power Level (mW)', fontsize=15)

def plot_slope_comparisons(datadir, dates, animal, lowess_frac):
    '''Plots the changes in slope throughout the day for each day against each other
    
    Arguments:
        dates: a list of the dates in string format
        animal: id number of the animal in string format 
        lowess_frac: how much you want to smooth the data by, 0-1 with 1 being completely and 0 being none
    
    Returns: 
        A plot of the slope comparisons between each day'''
    fig = plt.figure(1, figsize=(10, 4))
    for date in dates:
        df = pd.read_csv(os.path.join(f'{datadir}{date}-i{animal}', f'tr_info_dff_{date}.csv'))
        powers = sort_powers(df)
        success = df[df.trOutcome=='success']
        ys=[]
        for power in powers:
            ys.append(success.dffs[success.tLaserPowerMw==power].mean())
        slopes = [((ys[0]-0)/(powers[0]-0))]
        for i in range(0, len(powers)-1):
            slope = ((ys[i+1]-ys[i])/(powers[i+1]-powers[i]))
            slopes.append(slope)

    #     fig.add_subplot(1, 3, 1)
    #     plt.plot(powers, ys)
    #     plt.title('Average dF/F by Power Level')
    #     fig.add_subplot(1, 2, 1)

    #     plt.plot(powers, slopes)
    #     plt.title('Average Slope')
    #     fig.add_subplot(1, 2, 2)
        plt.plot(powers, lowess(slopes, powers, frac=lowess_frac, return_sorted=False))
    plt.title('Average Slopes')
    plt.xlabel('Power Level (mW)')
    plt.ylabel('Slope (dF/F per mW)')
    plt.legend(dates)
    plt.show()


def plot_cell_pop_avgs(datadir, date, animal, first_list, second_list, lowess_frac, scale=1, image_name='cell_pop_plot'):
    '''Plots the average df/fs for all of the single cells selected in a given day 
    
    Arguments:
        date: date the data was collected
        animal: id number of the animal
        first_list: the list df/fs designated for the first half of the day
        second_list: the list of df/fs designated for the second half of the day
        lowess_frac: amount of smoothing desired, between 0-1 with 0 being none
        scale: the amount the power is attenuated by if applicable
        image_name: the string name the plot will be saved under
        
    Returns:
        A plot showing the average single cell responses within the day and saves the plot. '''
    
    datadir_ = f'{datadir}{date}-i{animal}'
    tr_info = pd.read_csv(os.path.join(datadir_, f'tr_info-{date}-{animal}.csv'))
    index = int(tr_info[(tr_info.trOutcome == 'success')].shape[0]/2)
    halves = [first_list, second_list]
    powers = [np.unique(tr_info.tLaserPowerMw[tr_info.trOutcome =='success'][:index]), 
              np.unique(tr_info.tLaserPowerMw[tr_info.trOutcome =='success'][index:])]
    all_means = []
    all_sems = []
    for i, half in enumerate(halves):
        test = []
        for i in range(len(half[0])):
            for x in range(len(half)):
                test.append(half[x][i])

        means = []
        sems = []
        a = 0
        b = len(half)
        for i in range(len(half[0])):
            means.append(np.array(test[a:b]).mean())
            sems.append(np.array(test[a:b]).std()/np.sqrt(len(half)))
            a += len(half)
            b += len(half)
        all_means.append(means)
        all_sems.append(sems)
    plt.plot(np.array(powers[0])*scale, lowess(all_means[0], powers[0], frac=lowess_frac, return_sorted=False))
    plt.plot(np.array(powers[1])*scale, lowess(all_means[1], powers[1], frac=lowess_frac, return_sorted=False))
    plt.fill_between(np.array(powers[0])*scale, y1=np.array(lowess(all_means[0], powers[0], frac=lowess_frac, return_sorted=False))+np.array(all_sems[0]), 
                     y2=np.array(lowess(all_means[0], powers[0], frac=lowess_frac, return_sorted=False))-np.array(all_sems[0]), alpha=0.25)    
    plt.fill_between(np.array(powers[1])*scale, y1=np.array(lowess(all_means[1], powers[1], frac=lowess_frac, return_sorted=False))+np.array(all_sems[1]), 
                     y2=np.array(lowess(all_means[1], powers[1], frac=lowess_frac, return_sorted=False))-np.array(all_sems[1]), alpha=0.25) 
    plt.title('Average Response Across Single Cells')
    plt.ylabel('Average dF/F')
    plt.xlabel('Power Level (mW)')
    plt.savefig(f'{image_name}.png')
    plt.show()


def plot_smooth_comparisons(datadir, date, animal, image_name, blanking=True):
    '''Plot the differences in visualizing the df/f plot when the divisor is smoothed
    
    Arguments: 
        date: the date the data was collected
        animal: the id number of the animal
        image_name: the name the plots are saved under
    
    Returns:
        Four plots to show comparisons and saves it under image_name'''
    
    datadir_ = f'{datadir}{date}-i{animal}'
    tr_info = pd.read_csv(os.path.join(datadir_, f'tr_info-{date}-{animal}.csv'))
    tr_info = tr_info.drop('Unnamed: 0', axis=1)
    powers = tr_info.tLaserPowerMw.unique()[::-1]
    new_powers = sort_powers(tr_info)
    dFF = 0
    dFF_gaus = 0
    for val in new_powers[-1:]:
        power = val

        tifNums = tr_info[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)].index.tolist()
        tifNames = tr_info.tifNames[(tr_info.trOutcome == 'success') & (tr_info.tLaserPowerMw == power)]


        shapes = np.load(os.path.join(datadir_, 'i{}-{}-{}-nFrsTr.npy'.format(animal, date, power)))


        # Get Df/f info
        sub_info = tr_info.iloc[tifNums].reset_index()

        base_shift = 10 
        if blanking == True:
            res_shift = 0 
        if blanking == False:
            res_shift = 8

        stim_fr = sub_info.trHoldFrs - sub_info.trReactionTimeFrs
        base_fr = (stim_fr - base_shift).tolist()

        res_fr = (stim_fr + res_shift).tolist()
        stim_fr = stim_fr.tolist()

        im_stim = tifffile.imread(os.path.join(datadir_, f'renorm-mc-{power}.tif'))

        #df/f avg all trials
        past_fr = 0
        baseline = im_stim.min()

        sumFo = 0
        sumF = 0
        avg_pre = 10
        avg_post = 10
        tr_count = 0

        for file_index in range(len(base_fr)):
            if (base_fr[file_index] > 0):

                tr_count += 1

                sumFo += im_stim[past_fr + int(base_fr[file_index]): past_fr + int(base_fr[file_index]) + avg_pre].mean(axis = 0) - baseline

                sumF += im_stim[past_fr + int(res_fr[file_index]): past_fr + int(res_fr[file_index]) + avg_post].mean(axis = 0) - baseline

            past_fr += shapes[file_index]


        sumFo = sumFo / tr_count
        sumF = sumF / tr_count
        sumdFF_gaus = (sumF - sumFo) / gaussian_filter(sumFo, sigma=20)
        sumdFF = (sumF - sumFo)/sumFo
        dFF += sumdFF *100
        dFF_gaus += sumdFF_gaus*100
        df = sumF - sumFo
        smooth = gaussian_filter(sumFo, sigma = 50)


    dFF_plot = dFF # Lower
    fig = plt.figure(figsize=r_[1,0.75]*12, dpi=100)
    gs = mpl.gridspec.GridSpec(2,2)

    ax = fig.add_subplot(gs[0])
    im = plt.imshow(df, cmap = "RdBu_r")
    plt.clim(-500,500)
    plt.title(f'{date} df')
    plt.colorbar(im, label='dF')

    ax = fig.add_subplot(gs[1])
    im2 = plt.imshow(smooth, cmap='RdBu_r')
#     plt.clim(-1000,1000)
    plt.title(f'{date} smoother')
    plt.colorbar(im2, label='F')

    ax = fig.add_subplot(gs[2])
    im3 = plt.imshow(dFF, cmap='RdBu_r')
    plt.clim(-60,60)
    plt.title(f'{date} dF/F')
    plt.colorbar(im3, label='d F/F')

    ax = fig.add_subplot(gs[3])
    im4 = plt.imshow(dFF_gaus, cmap='RdBu_r')
    plt.clim(-60,60)
    plt.title(f'{date} dF/smoothed(F)')
    plt.colorbar(im4, label='d F/F')

    plt.savefig(f'{image_name}.png')
