import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import tifffile as tfl
from skimage import transform
from skimage.measure import block_reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import caiman as cm

import pytoolsMH as ptMH

from numpy import genfromtxt

import math
import cv2
from scipy import signal
import matplotlib.colors as mcolors
from matplotlib import animation
import matplotlib.cm as cm2
import time

import statistics

#import neo
from scipy.signal import find_peaks
from numpy import diff

#mworks imports
from mworksbehavior import mwkfiles
from mworksbehavior.imaging import intrinsic as ii
import mworksbehavior as mwb
import mworksbehavior.mwk_io
from glob import glob

import pandas as pd

sys.path.append('../src')
from functions import *

r_ = np.r_
a_ = np.asarray

### Functions ###

def get_data():
    # File Data
    root = input('Data Root Filename: ')
    date = input('Date: ')
    animal = input('Animal: ')
    path_to_data = root+date+'-'+animal+'/'
    
    print(f'Path to data: {path_to_data}')
    folder = input('Path to "reg_tif/img_file" (e.g. "suite_tifs/suite2p/plane0/": ')
    im_abc = input('ABC Filename: ')
    im_cba = input('CBA Filename: ')
    
    filepath2p = root+date+'-'+animal+'/'+folder
    full_path_abc = root+date+'-'+animal+'/'+folder+'reg_tif/'+im_abc
    full_path_cba = root+date+'-'+animal+'/'+folder+'reg_tif/'+im_cba
    
    
    # Pattern Data
    patternA_filename = input('Pattern A/1 Filename: ')
    patternB_filename = input('Pattern B/2 Filename: ')
    patternC_filename = input('Pattern C/3 Filename: ')
    
    return path_to_data, filepath2p, full_path_abc, full_path_cba, patternA_filename, patternB_filename, patternC_filename

def get_single_data():
    # File Data
    root = input('Data Root Filename: ')
    date = input('Date: ')
    animal = input('Animal: ')
    path_to_data = root+date+'-'+animal+'/'
    
    print(f'Path to data: {path_to_data}')
    folder = input('Path to "reg_tif/img_file" (e.g. "suite_tifs/suite2p/plane0/": ')
    im_a = input('A Filename: ')
    im_b = input('B Filename: ')
    im_c = input('C Filename: ')
    
    filepath2p = root+date+'-'+animal+'/'+folder
    full_path_a = root+date+'-'+animal+'/'+folder+'reg_tif/'+im_a
    full_path_b = root+date+'-'+animal+'/'+folder+'reg_tif/'+im_b
    full_path_c = root+date+'-'+animal+'/'+folder+'reg_tif/'+im_c
    
    
    # Pattern Data
    patternA_filename = input('Pattern A/1 Filename: ')
    patternB_filename = input('Pattern B/2 Filename: ')
    patternC_filename = input('Pattern C/3 Filename: ')
    
    return path_to_data, filepath2p, full_path_a, full_path_b, full_path_c, patternA_filename, patternB_filename, patternC_filename

def data_input():
        # Get Data
        path_to_data, filepath2p, im_abc, im_cba, patternA_filename, patternB_filename, patternC_filename = get_data()

        # Read in image files
        im_abc = tfl.imread(im_abc)
        im_cba = tfl.imread(im_cba)

        # Calculate df/f
        df_abc = im_abc - np.mean(im_abc[5:55,:,:], axis=0)
        dfof_abc = 100*df_abc/gaussian_filter(np.mean(im_abc[5:55,:,:], axis=0), sigma=20)

        df_cba = im_cba - np.mean(im_cba[5:55,:,:], axis=0)
        dfof_cba = 100*df_cba/gaussian_filter(np.mean(im_cba[5:55,:,:], axis=0), sigma=20)

        # Get Patterns
        zoom = 2
        radius_capture = 7.8
        cell_masks,nCell = generate_cell_masks_suite2p(filepath2p)

        # Pattern A
        cell_coords,patternA_coords = calculate_cell_pattern_coords(
            path_to_data, cell_masks, zoom, nCell, csv_pattern_filename=patternA_filename,pattern_template_units='micron')
        stim_iC_A,nostim_iC_A,all_iC = label_stimulated_cells(
            cell_coords, patternA_coords, zoom, nCell, radius_capture_microns=radius_capture)

        # Pattern B
        cell_coords,patternB_coords = calculate_cell_pattern_coords(
            path_to_data, cell_masks, zoom, nCell, csv_pattern_filename=patternB_filename,pattern_template_units='micron')
        stim_iC_B,nostim_iC_B,all_iC = label_stimulated_cells(
            cell_coords, patternB_coords, zoom, nCell, radius_capture_microns=radius_capture)

        # Pattern C
        cell_coords,patternC_coords = calculate_cell_pattern_coords(
            path_to_data, cell_masks, zoom, nCell, csv_pattern_filename=patternC_filename,pattern_template_units='micron')
        stim_iC_C,nostim_iC_C,all_iC = label_stimulated_cells(
            cell_coords, patternC_coords, zoom, nCell, radius_capture_microns=radius_capture)

        return dfof_abc, dfof_cba, cell_coords, patternA_coords, patternA_coords, patternB_coords, patternC_coords, stim_iC_A, stim_iC_B, stim_iC_C

def analysis_specifics():
        order = int(input('For Order ABC - CBA type 1, for CBA - ABC type 2: '))
        pattern = input('Pattern of interest (A/B/C): ')

        return order, pattern
    
def plot_dff_difference(dffs, zoom, stim_iC, cell_coords=None, 
                    radius_capture = 0, downscale_factor = 2, title=['Plot 1', 'Plot 2', 'Plot 1 - Plot 2'], save=None):

        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        ax = fig.add_subplot(gs[0])
        # abc
        im = plt.imshow(dffs[0],cmap='RdBu_r',vmin=-20, vmax = 20)
        plt.title(f'{title[0]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax1 = fig.add_subplot(gs[1])
        #cba
        im1 = plt.imshow(dffs[1],cmap='RdBu_r',vmin=-20, vmax = 20)
        plt.title(f'{title[1]}')
        plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax2 = fig.add_subplot(gs[2])
        #difference with stim pattern
        im2 = plt.imshow(dffs[0]-dffs[1],cmap='RdBu_r',vmin=-20, vmax = 20)
        plt.title(f'{title[2]}')
        plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        # plot pattern
        if cell_coords is not None:
            stim_coords = cell_coords[stim_iC, :]       
            for coords in stim_coords:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )

        if save:
            plt.savefig(f'{save}.png')
        plt.show()

def dff_difference_plots(set_data=True):

    dfof_abc, dfof_cba, cell_coords, patternA_coords, patternA_coords, patternB_coords, patternC_coords, stim_iC_A, stim_iC_B, stim_iC_C = data_input()
    
    order, pattern = analysis_specifics()
    if order==1:
        titles = [f'ABC: {pattern}', f'CBA: {pattern}', 'ABC - CBA']
        if pattern=='A':
            dffs = [dfof_abc[60],dfof_cba[62]]
            stim_iC = stim_iC_A
        elif pattern=='B':
            dffs = [dfof_abc[61],dfof_cba[61]]
            stim_iC = stim_iC_B
        elif pattern=='C':
            dffs = [dfof_abc[62],dfof_cba[60]]
            stim_iC = stim_iC_C
            
    elif order==2:
        titles = [f'CBA: {pattern}', f'ABC: {pattern}', 'CBA - ABC']
        if pattern=='C':
            dffs = [dfof_cba[60],dfof_abc[62]]
            stim_iC = stim_iC_C
        elif pattern=='B':
            dffs = [dfof_cba[61],dfof_abc[61]]
            stim_iC = stim_iC_B
        elif pattern=='A':
            dffs = [dfof_cba[62],dfof_abc[60]]
            stim_iC = stim_iC_A

    save_name = input('Save plot as: ')
    
    plot_dff_difference(dffs=dffs, zoom=2, stim_iC=stim_iC, cell_coords=cell_coords, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)
    
    rerun = input('Would you like to make another plot? (y/n): ')
    if rerun=='y':

        while rerun=='y':
            order, pattern = analysis_specifics() 
            if order==1:
                titles = [f'ABC: {pattern}', f'CBA: {pattern}', 'ABC - CBA']
                if pattern=='A':
                    dffs = [dfof_abc[60],dfof_cba[62]]
                    stim_iC = stim_iC_A
                elif pattern=='B':
                    dffs = [dfof_abc[61],dfof_cba[61]]
                    stim_iC = stim_iC_B
                elif pattern=='C':
                    dffs = [dfof_abc[62],dfof_cba[60]]
                    stim_iC = stim_iC_C

            elif order==2:
                titles = [f'CBA: {pattern}', f'ABC: {pattern}', 'CBA - ABC']
                if pattern=='C':
                    dffs = [dfof_cba[60],dfof_abc[62]]
                    stim_iC = stim_iC_C
                elif pattern=='B':
                    dffs = [dfof_cba[61],dfof_abc[61]]
                    stim_iC = stim_iC_B
                elif pattern=='A':
                    dffs = [dfof_cba[62],dfof_abc[63]]
                    stim_iC = stim_iC_A

            save_name = input('Save plot as: ')
            plot_dff_difference(dffs=dffs, zoom=2, stim_iC=stim_iC, cell_coords=cell_coords, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)
            rerun = input('Would you like to make another plot? (y/n): ')
        
    elif rerun=='n':
        print('Done.')

def dff_stimpat():
    dfof_abc, dfof_cba, cell_coords, patternA_coords, patternA_coords, patternB_coords, patternC_coords, stim_iC_A, stim_iC_B, stim_iC_C = data_input()
    zoom=2
    print('Plotting sequence A-B-C laser pattern')
    save_name = input('Save plot as: ')            

    dff=dfof_abc
    stim_iC=[stim_iC_A, stim_iC_B, stim_iC_C]
    frames=[60, 61, 62]
    titles=['A', 'B', 'C']
    pattern_coords=[patternA_coords, patternB_coords, patternC_coords]
    cell_coord=None

    save_name=save_name
    plot_dff_stimpat(dff, zoom, stim_iC, frames, pattern_coords, cell_coord, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)

    print('Plotting sequence A-B-C stimulated cells')
    save_name = input('Save plot as: ') 
    pattern_coords=None
    cell_coord=cell_coords

    save_name=save_name
    plot_dff_stimpat(dff, zoom, stim_iC, frames, pattern_coords, cell_coord, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)

    print('Plotting sequence C-B-A laser pattern')
    save_name = input('Save plot as: ')            

    dff=dfof_cba
    stim_iC=[stim_iC_C, stim_iC_B, stim_iC_A]
    frames=[60, 61, 62]
    titles=['C', 'B', 'A']
    pattern_coords=[patternC_coords, patternB_coords, patternA_coords]
    cell_coord=None

    save_name=save_name
    plot_dff_stimpat(dff, zoom, stim_iC, frames, pattern_coords, cell_coord, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)

    print('Plotting sequence C-B-A stimulated cells')
    save_name = input('Save plot as: ') 
    pattern_coords=None
    cell_coord=cell_coords

    save_name=save_name
    plot_dff_stimpat(dff, zoom, stim_iC, frames, pattern_coords, cell_coord, radius_capture=7.8, downscale_factor=2, title=titles, save=save_name)

def plot_dff_stimpat(dff=None, zoom=None, stim_iC=None, frames=[61, 62, 63], 
                     pattern_coords=None, cell_coords=None, 
                     radius_capture = 10, downscale_factor=2, title=['Plot 1', 'Plot 2', 'Plot 3'], save=None):

    conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(nrows=1,ncols=3)
    
    ax = fig.add_subplot(gs[0])
    im = plt.imshow(dff[frames[0]],cmap='RdBu_r',vmin=-20, vmax = 20)
    plt.title(f'{title[0]}')
    plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    ax1 = fig.add_subplot(gs[1])
    im1 = plt.imshow(dff[frames[1]],cmap='RdBu_r',vmin=-20, vmax = 20)
    plt.title(f'{title[1]}')
    plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    ax2 = fig.add_subplot(gs[2])
    im2 = plt.imshow(dff[frames[2]],cmap='RdBu_r',vmin=-20, vmax = 20)
    plt.title(f'{title[2]}')
    plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    # plot pattern
    if pattern_coords is not None:
        for coords in pattern_coords[0]:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[1]:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[2]:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )            
            
    if cell_coords is not None:
        stim_coords = cell_coords[stim_iC[0], :]       
        for coords in stim_coords:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
        stim_coords = cell_coords[stim_iC[1], :]       
        for coords in stim_coords:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
        stim_coords = cell_coords[stim_iC[2], :]       
        for coords in stim_coords:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

    if save:
        plt.savefig(f'{save}.png')
    plt.show()

    
def plot_single_difference(dffs, zoom, stim_iC, vmin, vmax, pattern_coords=None, cell_coords=None, 
                    radius_capture = 0, downscale_factor = 2, title=['Plot 1', 'Plot 2', 'Plot 1 - Plot 2'], save=None):

        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        ax = fig.add_subplot(gs[0])
        # abc
        im = plt.imshow(dffs[0],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[0]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax1 = fig.add_subplot(gs[1])
        #cba
        im1 = plt.imshow(dffs[1],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[1]}')
        plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax2 = fig.add_subplot(gs[2])
        #difference with stim pattern
        im2 = plt.imshow(dffs[0]-dffs[1],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[2]}')
        plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        # plot pattern
        if pattern_coords is not None:
            for coords in pattern_coords:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )

        
        if cell_coords is not None:
            stim_coords = cell_coords[stim_iC, :]       
            for coords in stim_coords:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )

        if save:
            plt.savefig(f'{save}.png')
        plt.show()
        
def plot_dff_stimpat(dff=None, zoom=None, vmin=-30, vmax=30, stim_iC=None, frames=[61, 62, 63], 
                     pattern_coords=None, cell_coords=None, 
                     radius_capture = 0, downscale_factor=2, title=['Plot 1', 'Plot 2', 'Plot 3'], save=None):

    conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(nrows=1,ncols=3)
    
    ax = fig.add_subplot(gs[0])
    im = plt.imshow(dff[frames[0]],cmap='RdBu_r',vmin=-50, vmax = 50)
    plt.title(f'{title[0]}')
    plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    ax1 = fig.add_subplot(gs[1])
    im1 = plt.imshow(dff[frames[1]],cmap='RdBu_r',vmin=-50, vmax = 50)
    plt.title(f'{title[1]}')
    plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    ax2 = fig.add_subplot(gs[2])
    im2 = plt.imshow(dff[frames[2]],cmap='RdBu_r',vmin=-50, vmax = 50)
    plt.title(f'{title[2]}')
    plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')

    # plot pattern
    if pattern_coords is not None:
        for coords in pattern_coords[0]:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[1]:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[2]:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )            
            
    if cell_coords is not None:
        stim_coords = cell_coords[stim_iC[0], :]       
        for coords in stim_coords:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
        stim_coords = cell_coords[stim_iC[1], :]       
        for coords in stim_coords:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
        stim_coords = cell_coords[stim_iC[2], :]       
        for coords in stim_coords:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

    if save:
        plt.savefig(f'{save}.png')
    plt.show()

    
def plot_dff_difference(dffs, zoom, stim_iC, vmin, vmax, cell_coords=None, 
                    radius_capture = 0, downscale_factor = 2, title=['Plot 1', 'Plot 2', 'Plot 1 - Plot 2'], save=None):

        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        ax = fig.add_subplot(gs[0])
        # abc
        im = plt.imshow(dffs[0],cmap='RdBu_r',vmin=vmin, vmax =vmax)
        plt.title(f'{title[0]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax1 = fig.add_subplot(gs[1])
        #cba
        im1 = plt.imshow(dffs[1],cmap='RdBu_r',vmin=vmin, vmax = vmax)
        plt.title(f'{title[1]}')
        plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax2 = fig.add_subplot(gs[2])
        #difference with stim pattern
        im2 = plt.imshow(dffs[0]-dffs[1],cmap='RdBu_r',vmin=vmin, vmax = vmax)
        plt.title(f'{title[2]}')
        plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        # plot pattern
        if cell_coords is not None:
            stim_coords = cell_coords[stim_iC, :]       
            for coords in stim_coords:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )

        if save:
            plt.savefig(f'{save}.png')
        plt.show()
        
def plot_dff_pat(dffs, vmin, vmax, zoom, stim_iC, pattern_coords=None, cell_coords=None, 
                    radius_capture = 0, downscale_factor = 2, title=['Plot 1', 'Plot 2', 'Plot 1 - Plot 2'], save=None):

        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        ax = fig.add_subplot(gs[0])

        im = plt.imshow(dffs[0],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[0]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')

        ax1 = fig.add_subplot(gs[1])

        im1 = plt.imshow(dffs[1],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[1]}')
        plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')
        
        ax2 = fig.add_subplot(gs[2])

        im1 = plt.imshow(dffs[2],cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[2]}')
        plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')
        
        if pattern_coords is not None:
            for coords in pattern_coords[0]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
            for coords in pattern_coords[1]:
                ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
            for coords in pattern_coords[2]:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )            

        # plot pattern
        if cell_coords is not None:
            stim_coords_a = cell_coords[stim_iC[0], :]       
            for coords in stim_coords_a:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

            stim_coords_b = cell_coords[stim_iC[1], :]       
            for coords in stim_coords_b:
                ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

            stim_coords_c = cell_coords[stim_iC[2], :]       
            for coords in stim_coords_c:
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )


        if save:
            plt.savefig(f'{save}.png')
        plt.show()
        
        
def plot_whole_pat(dffs, vmin, vmax, zoom, stim_iC, pattern_coords=None, cell_coords=None, cell_masks=None,
                    radius_capture = 0, downscale_factor = 2, title=['Plot 1', 'Plot 2', 'Plot 1 - Plot 2'], save=None):

        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(nrows=1,ncols=3)
        ax = fig.add_subplot(gs[0])

        im = plt.imshow(dffs,cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[0]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')
        
        ax1 = fig.add_subplot(gs[1])

        im = plt.imshow(dffs,cmap='RdBu_r',vmin=vmin, vmax=vmax)
        plt.title(f'{title[1]}')
        plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
        plt.axis('off')
                
        ax2 = fig.add_subplot(gs[2])
        cell_masks = cell_masks.sum(axis=2)
        im = plt.imshow(cell_masks,cmap='Greys')
        plt.title(f'{title[2]}')
        plt.axis('off')


        if pattern_coords is not None:
            for coords in pattern_coords[0]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='red',ls='--',lw=1,alpha=1) )
            
            for coords in pattern_coords[1]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='red',ls='--',lw=1,alpha=1) )
            
            for coords in pattern_coords[2]:
                ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )            
                ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='red',ls='--',lw=1,alpha=1) )

        # plot pattern
        if cell_coords is not None:
            stim_coords_a = cell_coords[stim_iC[0], :]       
            for coords in stim_coords_a:
                ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

            stim_coords_b = cell_coords[stim_iC[1], :]       
            for coords in stim_coords_b:
                ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

            stim_coords_c = cell_coords[stim_iC[2], :]       
            for coords in stim_coords_c:
                ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
        
        # if stim_masks is not None:
        #     ax2.imshow(cell_masks,vmin=0,vmax=1,cmap='Greys', alpha=0.2)
        #     ax2.imshow(stim_masks[1],vmin=0,vmax=1,cmap='Greys', alpha=0.2)
        #     ax2.imshow(stim_masks[2],vmin=0,vmax=1,cmap='Greys', alpha=0.2)

        if save:
            plt.savefig(f'{save}.png')
        plt.show()
        
def plot_patterns(dffs, zoom, stim_iC, vmin=-10, vmax=10, pattern_coords=None, cell_coords=None, radius_capture=7.8, downscale_factor=2, title=['A','B','C'], save=None):
    conversion_factor = ((512/downscale_factor)/(1037/zoom))
    
    fig = plt.figure(figsize=(20,12))
    gs = fig.add_gridspec(nrows=1, ncols=3)
    ax = fig.add_subplot(gs[0])
    # Pattern A
    im = plt.imshow(dffs[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'{title[0]}')
    plt.colorbar(im, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')
    
    ax1 = fig.add_subplot(gs[1])
    # Pattern B
    im1 = plt.imshow(dffs[1], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'{title[1]}')
    plt.colorbar(im1, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')
    
    ax2 = fig.add_subplot(gs[2])
    # Pattern C
    im2 = plt.imshow(dffs[2], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'{title[2]}')
    plt.colorbar(im2, label='d F/F', fraction=0.046, pad=0.04)
    plt.axis('off')
    
    if pattern_coords is not None:
        for coords in pattern_coords[0]:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[1]:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )
        for coords in pattern_coords[2]:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='black',ls='--',lw=1,alpha=1) )            

    # plot pattern
    if cell_coords is not None:
        stim_coords_a = cell_coords[stim_iC[0], :]       
        for coords in stim_coords_a:
            ax.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

        stim_coords_b = cell_coords[stim_iC[1], :]       
        for coords in stim_coords_b:
            ax1.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )

        stim_coords_c = cell_coords[stim_iC[2], :]       
        for coords in stim_coords_c:
            ax2.add_patch( Circle((coords[0],coords[1]),radius=radius_capture*conversion_factor,fill=False,ec='blue',ls='--',lw=1,alpha=1) )
    
    if save is not None:
        plt.savefig(f'{save}')
    plt.show()
    

    
def generate_cell_masks(cell_masks, zoom, stim_ic):
    masks = np.sum(cell_masks,axis=2)
    stim_masks_A = np.sum(cell_masks[:,:,stim_ic[0]],axis=2)
    stim_masks_B = np.sum(cell_masks[:,:,stim_ic[1]],axis=2)
    stim_masks_C = np.sum(cell_masks[:,:,stim_ic[2]],axis=2)

    masks[masks>0] = 1
    masks[masks==0] = np.nan

    stim_masks_A[stim_masks_A>0] = 1
    stim_masks_A[stim_masks_A==0] = np.nan

    stim_masks_B[stim_masks_B>0] = 1
    stim_masks_B[stim_masks_B==0] = np.nan

    stim_masks_C[stim_masks_C>0] = 1
    stim_masks_C[stim_masks_C==0] = np.nan

    return masks, stim_masks_A, stim_masks_B, stim_masks_C


def overlay_cellmask(file_path, file_nm, stim_masks, pattern=None, save=None, vmin=None, vmax=None):
    im_frame = tfl.imread(file_nm)
    fig = plt.figure(figsize=[6.922,6.922],frameon=False)
    plt.imshow(im_frame/np.mean(im_frame), cmap='gray', vmin=.999, vmax=1.005)#, interpolation='none', vmin=0, vmax=255)
    if pattern == 'A':
        plt.imshow(stim_masks[0],vmin=0,vmax=1,cmap='viridis')
    elif pattern == 'B':
        plt.imshow(stim_masks[1],vmin=0,vmax=1,cmap='winter')
    elif pattern == 'C':
        plt.imshow(stim_masks[2],vmin=0,vmax=1,cmap='cool')
    else:
        plt.imshow(stim_masks[0],vmin=0,vmax=1,cmap='viridis')
        plt.imshow(stim_masks[1],vmin=0,vmax=1,cmap='winter')
        plt.imshow(stim_masks[2],vmin=0,vmax=1,cmap='cool')
    if save:
        plt.savefig(os.path.join(file_path,f'{save}.png'),bbox_inches='tight')
        
#------------------------------------------------------------------------------
# Traces Functions

def select_traces(filepath2p, path_to_image):
    im_file = tfl.imread(path_to_image)
    
    isCell = np.load(filepath2p+'iscell.npy',allow_pickle=True)
    F0 = np.load(filepath2p+'F.npy',allow_pickle=True)
    fneu = np.load(filepath2p+'Fneu.npy',allow_pickle=True)
    # F = F0 - fneu

    iscell_accepted_idx = np.where(isCell[:,0]==1)[0]

    raw_traces = F0[iscell_accepted_idx]
    
    return raw_traces, im_file

def plot_seq_traces_A(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2, ax3) = plt.subplots(figsize=np.r_[3,4]*4,nrows=3,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][small].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][large].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    else:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)       
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average Stimulated Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(55, 70)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    ax3.plot(x, mean)
    ax3.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax3.plot(x_shift, mean_c)
    ax3.fill_between(x_shift, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax3.set_title('Aligned Stimulated Cell Traces')
    ax3.legend(labels = ['ABC', 'CBA'])
    ax3.set_xlim(55, 70)
    ax3.set_ylabel('dF/F (%)')
    ax3.set_xlabel('Frame')
    
    if save:
        plt.savefig(f"{save}.png")
    return small, small_c, large, large_c

def plot_seq_traces_B(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2) = plt.subplots(figsize=np.r_[3,3]*4,nrows=2,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][small].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][large].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    else:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)        
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average Stimulated Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(55,70)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    if save:
        plt.savefig(f"{save}.png")
    
    return small, small_c, large, large_c

def plot_seq_traces_C(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2, ax3) = plt.subplots(figsize=np.r_[3,4]*4,nrows=3,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][small].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][large].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)
    
    else:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(55,70)       
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average Stimulated Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(55,70)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    ax3.plot(x_shift, mean)
    ax3.fill_between(x_shift, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax3.plot(x, mean_c)
    ax3.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax3.set_title('Aligned Stimulated Cell Traces')
    ax3.legend(labels = ['ABC', 'CBA'])
    ax3.set_xlim(55,70)
    ax3.set_ylabel('dF/F (%)')
    ax3.set_xlabel('Frame')
    
    if save:
        plt.savefig(f"{save}.png")
    return small, small_c, large, large_c

def plot_seq_traces_all_stim(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2) = plt.subplots(figsize=np.r_[3,3]*4,nrows=2,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][small].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(50,90)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:][large].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(50,90)
    
    else:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T, color='C0')
        ax1.plot(dffs[1][stim_pattern,:].T, color='C1')
        ax1.set_title('Stimulated Cell Traces')
        ax1.set_xlim(50,90)         
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average Stimulated Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(50, 90)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    if save:
        plt.savefig(f"{save}.png")
    return small, small_c, large, large_c

def plot_seq_traces_all(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2) = plt.subplots(figsize=np.r_[3,3]*4,nrows=2,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T)
        ax1.plot(dffs[1][stim_pattern,:][small].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T)
        ax1.plot(dffs[1][stim_pattern,:][large].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)
    
    elif selection is None:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T)
        ax1.plot(dffs[1][stim_pattern,:].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)         
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average All Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(50, 90)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    if save:
        plt.savefig(f"{save}.png")
    return small, small_c, large, large_c

def plot_seq_traces_no_stim(dffs, stim_pattern, selection=None, save=None):
    small = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) < 5)
    large = np.where(dffs[0][stim_pattern,60:70].mean(axis=1) >= 5)
    small_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) < 5)
    large_c = np.where(dffs[1][stim_pattern,60:70].mean(axis=1) >= 5)
    
    fig,(ax1,ax2) = plt.subplots(figsize=np.r_[3,3]*4,nrows=2,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    
    if selection == 'small':
        sem = dffs[0][stim_pattern,:][small].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][small].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][small_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][small_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][small].T)
        ax1.plot(dffs[1][stim_pattern,:][small].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)
    
    elif selection == 'large':
        sem = dffs[0][stim_pattern,:][large].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:][large].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:][large_c].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:][large_c].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:][large].T)
        ax1.plot(dffs[1][stim_pattern,:][large].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)
    
    else:
        sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean = dffs[0][stim_pattern,:].mean(axis=0)

        sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
        mean_c = dffs[1][stim_pattern,:].mean(axis=0)

        ax1.plot(dffs[0][stim_pattern,:].T)
        ax1.plot(dffs[1][stim_pattern,:].T)
        ax1.set_title('All Cell Traces')
        ax1.set_xlim(50,90)         
    
    ax2.plot(x, mean)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    ax2.plot(x, mean_c)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    ax2.set_title('Average All Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(50, 90)
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    if save:
        plt.savefig(f"{save}.png")

def plot_sequence(dffs, stim_ic, xlims, Acolor, Bcolor, Ccolor):
    fig,(ax1) = plt.subplots(figsize=np.r_[2,2]*2,nrows=1,ncols=1)
    x = np.linspace(0,dffs.shape[1], num=dffs.shape[1]) 

    std_0 = dffs[stim_ic[0],:].std()/np.sqrt(dffs.shape[0])
    mean_0 = dffs[stim_ic[0],:].mean(axis=0)
    std_1 = dffs[stim_ic[1],:].std()/np.sqrt(dffs.shape[0])
    mean_1 = dffs[stim_ic[1],:].mean(axis=0)
    std_2 = dffs[stim_ic[2],:].std()/np.sqrt(dffs.shape[0])
    mean_2 = dffs[stim_ic[2],:].mean(axis=0)
        
    ax1 = plt.gca()
    ax1.tick_params(direction='out', length=1, pad=2, width=.25)
    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(0.25)
    plt.tight_layout() 
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    plt.plot(x, mean_0, c=Acolor)
    plt.fill_between(x, y1=mean_0+std_0, y2=mean_0-std_0, alpha=0.25, color=Acolor)
    plt.plot(x, mean_1, c=Bcolor)
    plt.fill_between(x, y1=mean_1+std_1, y2=mean_1-std_1, alpha=0.25, color=Bcolor)
    plt.plot(x, mean_2, c=Ccolor)
    plt.fill_between(x, y1=mean_2+std_2, y2=mean_2-std_2, alpha=0.25, color=Ccolor)
    plt.xlim(xlims[0], xlims[1])
    plt.legend(labels=['A', 'B', 'C'])
    plt.xlabel('Frame')
    plt.ylabel('dF/F (%)')
    
def plot_A_traces(dffs, stim_pattern, xlim1, xlim2, Acolor='C0', Ccolor='C1', save=None):
    
    fig,(ax2) = plt.subplots(figsize=np.r_[2,2]*2,nrows=1,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])

    sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean = dffs[0][stim_pattern,:].mean(axis=0)

    sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean_c = dffs[1][stim_pattern,:].mean(axis=0)   
    
    ax2 = plt.gca()
    ax2.tick_params(direction='out', length=1, pad=2, width=.25)
    for axis in ['bottom','left']:
        ax2.spines[axis].set_linewidth(0.25)
    plt.tight_layout() 
    
    ax2.plot(x, mean, color=Acolor)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25, color=Acolor)
    ax2.plot(x, mean_c, color=Ccolor)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25, color=Ccolor)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Pattern A Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(xlim1,xlim2) 
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    # ax3.plot(x, mean)
    # ax3.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25)
    # ax3.plot(x_shift, mean_c)
    # ax3.fill_between(x_shift, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25)
    # ax3.set_title('Aligned Stimulated Cell Traces')
    # ax3.legend(labels = ['ABC', 'CBA'])
    # ax3.set_xlim(xlim1,xlim2) 
    # ax3.set_ylabel('dF/F (%)')
    # ax3.set_xlabel('Frame')
    plt.axvline(58, color='navy', ls='--', lw=0.7)
    plt.axvline(60, color='green', ls='--', lw=0.7)
    
    if save:
        plt.savefig(f"{save}.pdf", dpi=300)
        
def plot_B_traces(dffs, stim_pattern, xlim1, xlim2, Acolor='C0', Ccolor='C1', save=None):

    fig,(ax2) = plt.subplots(figsize=np.r_[2,2]*2,nrows=1,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])

    sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean = dffs[0][stim_pattern,:].mean(axis=0)

    sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean_c = dffs[1][stim_pattern,:].mean(axis=0)      
        
    ax2 = plt.gca()
    ax2.tick_params(direction='out', length=1, pad=2, width=.25)
    for axis in ['bottom','left']:
        ax2.spines[axis].set_linewidth(0.25)
    plt.tight_layout() 
    
    ax2.plot(x, mean, color=Acolor)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25, color=Acolor)
    ax2.plot(x, mean_c, color=Ccolor)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25, color=Ccolor)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Pattern B Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(xlim1, xlim2)  
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    plt.axvline(59, color='navy', ls='--', lw=0.8)
    plt.axvline(59, color='green', ls='--', lw=0.7)
    
    if save:
        plt.savefig(f"{save}.pdf", dpi=300)


def plot_C_traces(dffs, stim_pattern, xlim1, xlim2, Acolor='C0', Ccolor='C1', save=None):
    
    fig,(ax2) = plt.subplots(figsize=np.r_[2,2]*2,nrows=1,ncols=1)
    x = np.linspace(0,dffs[0].shape[1], num=dffs[0].shape[1]) 
    x_shift = np.linspace(-2,dffs[0].shape[1], num=dffs[0].shape[1])

    sem = dffs[0][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean = dffs[0][stim_pattern,:].mean(axis=0)

    sem_c = dffs[1][stim_pattern,:].std()/np.sqrt(stim_pattern.shape[0])
    mean_c = dffs[1][stim_pattern,:].mean(axis=0)   
    
    ax2 = plt.gca()
    ax2.tick_params(direction='out', length=1, pad=2, width=.25)
    for axis in ['bottom','left']:
        ax2.spines[axis].set_linewidth(0.25)
    plt.tight_layout() 
    
    ax2.plot(x, mean, color=Acolor)
    ax2.fill_between(x, y1=mean+sem, y2=mean-sem, alpha=0.25, color=Acolor)
    ax2.plot(x, mean_c, color=Ccolor)
    ax2.fill_between(x, y1=mean_c+sem_c, y2=mean_c-sem_c, alpha=0.25, color=Ccolor)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Pattern C Cell Traces')
    ax2.legend(labels = ['ABC', 'CBA'])
    ax2.set_xlim(xlim1,xlim2) 
    ax2.set_ylabel('dF/F (%)')
    ax2.set_xlabel('Frame')

    plt.axvline(60, color='navy', ls='--', lw=0.7)
    plt.axvline(58, color='green', ls='--', lw=0.7)
    
    if save:
        plt.savefig(f"{save}.pdf", dpi=300, bbox_inches='tight')

def plot_ABC(dffs, stim_ic, xlims, Acolor, Bcolor, Ccolor, sequence):
    fig,(ax1) = plt.subplots(figsize=np.r_[2,2]*2,nrows=1,ncols=1)
    x = np.linspace(0,dffs.shape[1], num=dffs.shape[1]) 
    
    std_0 = dffs[stim_ic[0],:].std()/np.sqrt(dffs.shape[0])
    mean_0 = dffs[stim_ic[0],:].mean(axis=0)
    std_1 = dffs[stim_ic[1],:].std()/np.sqrt(dffs.shape[0])
    mean_1 = dffs[stim_ic[1],:].mean(axis=0)
    std_2 = dffs[stim_ic[2],:].std()/np.sqrt(dffs.shape[0])
    mean_2 = dffs[stim_ic[2],:].mean(axis=0)
    
    ax1 = plt.gca()
    ax1.tick_params(direction='out', length=1, pad=2, width=.25)
    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(0.25)
    plt.tight_layout() 
    
    ax1.plot(x, mean_0, color=Acolor)
    ax1.fill_between(x, y1=mean_0+std_0, y2=mean_0-std_0, alpha=0.25, color=Acolor)
    ax1.plot(x, mean_1, color=Bcolor)
    ax1.fill_between(x, y1=mean_1+std_1, y2=mean_1-std_1, alpha=0.25, color=Bcolor)
    ax1.plot(x, mean_2, color=Ccolor)
    ax1.fill_between(x, y1=mean_2+std_2, y2=mean_2-std_2, alpha=0.25, color=Ccolor)
    plt.xlim(xlims[0], xlims[1])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    if sequence == 'ABC':
        plt.axvline(58, color=Acolor, ls='--', lw=0.7)
        plt.axvline(59, color=Bcolor, ls='--', lw=0.7)
        plt.axvline(60, color=Ccolor, ls='--', lw=0.7)
    elif sequence == 'CBA':
        plt.axvline(60, color=Acolor, ls='--', lw=0.7)
        plt.axvline(59, color=Bcolor, ls='--', lw=0.7)
        plt.axvline(58, color=Ccolor, ls='--', lw=0.7)        
    plt.legend(labels=['Pattern A', 'Pattern B', 'Pattern C'])
    plt.xlabel('Frame')
    plt.ylabel('dF/F (%)')
    plt.savefig(f"/Users/deveauce/Data/PaperFigures/{sequence}.pdf", dpi=300, bbox_inches='tight')


def deconvolve(traces, trial_params, sequence=None):
    F = traces
    deconvF = np.zeros(F.shape)
    calciumEst = np.zeros(F.shape)
    for c in np.arange(0,F.shape[0]):
        if np.sum(F[c,:])!=0:
            c0, b1, c1, g, sn, sp, lam = cm.source_extraction.cnmf.deconvolution.constrained_foopsi(F[c,:],p=1)
            deconvF[c,:] = sp
            calciumEst[c,:] = c0
        else:
            print('cell '+np.str(c)+' skipped')
    deconvF_reshape = np.reshape(deconvF,(deconvF.shape[0],trial_params['nTrial']*2,trial_params['nFrTrial']))
    if sequence == 'ABC':
        deconvF_mean = deconvF_reshape[:, :trial_params['nTrial'], :].mean(axis=1)
    elif sequence == 'CBA':
        deconvF_mean = deconvF_reshape[:, trial_params['nTrial']:, :].mean(axis=1)
    else:
        deconvF_mean = deconvF_reshape[:, :, :].mean(axis=1)
    return deconvF_mean