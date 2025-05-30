# import and load modules
import numpy as np; import pandas as pd; import math as math; import copy
from scipy.ndimage import gaussian_filter as gaussian_filter; import scipy.stats as stats
import statsmodels.api as sm; lowess = sm.nonparametric.lowess; from skimage import transform

#mworks imports
from mworksbehavior import mwkfiles; from mworksbehavior.imaging import intrinsic as ii; import mworksbehavior as mwb; import mworksbehavior.mwk_io

import matplotlib as mpl; import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec; from matplotlib import cm; from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable #from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle; from matplotlib.patches import Rectangle

import os, sys; from glob import glob; from pathlib import Path; import pickle
import tifffile as tfl; from datetime import datetime

#local files and shorthand notations for assigning common functions to variable names
r_ = np.r_; a_ = np.asarray; n_ = np.newaxis

#### utility class for handling caiman output
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
def load_npys(dir):
    """Load all NumPy arrays found in the specified directory as variables associated with their file names.
    Params:directory (str): The path to the directory containing the NumPy files.
    Returns: dict where keys are file names (without extensions) and values are the loaded NumPy arrays."""
    npy_arrs = {}
    
    for filename in os.listdir(dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir, filename)
            arr = np.load(file_path)
            variable_name = os.path.splitext(filename)[0]
            npy_arrs[variable_name] = arr
            
    return npy_arrs 

def calc_coords_multipat(nCell,zoom,file_path_pattern,cell_masks,csv_list=['1.csv'],downscale_factor=2):
        '''calculate coordinates of cell mask center of masses and coordinates of stim pattern targets'''
        # get coordinates of center of masses of masks in pixels
        cell_coords = []
        for iC in range(nCell):
            ys,xs = np.where(cell_masks[:,:,iC]>0)
            coord = [np.average(xs),np.average(ys)]
            cell_coords.append(coord)
        cell_coords = np.asarray(cell_coords)
        pat_coords_all = []
        for i in range(len(csv_list)): # get coords of pattern targets in microns
            pattern_file = os.path.join(file_path_pattern,csv_list[i])
            df = pd.read_csv(pattern_file)
            coords = np.rollaxis(np.asarray((df['X'],df['Y'])),1,0)
            # convert micron->pxl (hard-coded in SI defaults) and scale to match downscaled image
            conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pxl/micrn
            pix_coords = coords*conversion_factor
            pattern_coords = pix_coords[1:] # remove first entry - zero order position
            pat_coords_all.append(pattern_coords)
            
        return cell_coords,pat_coords_all,conversion_factor
    
def label_stimd_cells_multipat(nCell,zoom,cell_coords,pat_coords_all,radius_microns=10,downscale_factor=2):
        '''compare coords of pattern targets and cells to find which cells are directly stimulated 
        within a radius of the pattern target coordinates'''
        print('Finding cell IDs within direct stimulation radius...', end=' ')
        # convert radius_microns to radius_pix
        conversion_factor = ((512/downscale_factor)/(1037/zoom)) # units of pixel/micron
        radius_pix = radius_microns*conversion_factor
        # find within-radius cell_coords
        stim_iC = []
        for iPat in range(len(pat_coords_all)):
            stim_iC_indivPat = []
            for iPC in range(pat_coords_all[iPat].shape[0]):
                pattern_coord = pat_coords_all[iPat][iPC,:]
                euclidean_distances = np.linalg.norm(cell_coords-pattern_coord,axis=1)
                within_radius_iC = np.where(euclidean_distances<radius_pix)[0]
                [stim_iC_indivPat.append(iC) for iC in within_radius_iC]
            stim_iC.append(stim_iC_indivPat)
        # get unique values from list as stimulated cell indices  
        stim_iC_unique = [item for sublist in stim_iC for item in sublist] 
        stim_iC_unique = list(set(item for sublist in stim_iC for item in sublist)) #flatten list using set fucntion then convert back to list
        print('Stimulated cell IDs found.')
        # annotate the cells which are stimulated vs. non-stimulated
        all_iC = list(range(nCell))
        nostim_iC = [iC for iC in all_iC if iC not in stim_iC]
        return stim_iC,stim_iC_unique,nostim_iC,all_iC
        
        
