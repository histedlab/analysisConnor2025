import numpy as np; import pandas as pd
import scipy.io
from skimage import io, transform
from scipy.ndimage import gaussian_filter as gaussian_filter

import matplotlib as mpl; import matplotlib.pyplot as plt
from matplotlib import cm; from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

import os, sys; from glob import glob; from pathlib import Path
from datetime import datetime; import tifffile as tfl
import caiman as cm; import pytoolsMH as ptMH # import custom packages

# from caiman.utils.visualization import get_contours
sys.path.append(os.path.expanduser('~/Repositories/CaImAn/caiman/source_extraction/cnmf/'))
from deconvolution import *

# these are shorthand notations for assigning common functions to variable names
r_ = np.r_
a_ = np.asarray

#### utility class for handling caiman output
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
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
    with tfl.TiffFile(infile) as tif:
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
            chunk = tfl.imread(infile,key=ix)
            print(fr+nFrChunk, end=' ')
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        if fr+nFrChunk > T:
            ix = r_[fr:T]
            chunk = tfl.imread(infile, key=ix)
            print(T)
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        del chunk

    im = im.astype('int16')
    tfl.imsave(outfile, im, bigtiff=True)
    print('done. saved to {}'.format(outfile))
    
    
def makePositive(im,file_path):
    #tfl.imsave(outfile_stim,im,bigtiff=True)
    outfile_stim_bg = os.path.join(file_path,'imageFrames_bgSub.tif')
    outfile_resave = os.path.join(file_path,'batch_resave.tif')

    print("Computing min...", datetime.now().time())
    im_bg = np.min(np.mean(im, axis=0)) # shape is fr,side,side
    im = im - im_bg
    print("Background subtracted. bg was", im_bg, datetime.now().time())
    im[im<0] = 0
    tfl.imwrite(outfile_stim_bg,im,bigtiff=True)
    os.remove(outfile_resave)
    #os.remove(outfile_stim)
    print('Infile preprocessed and saved at', datetime.now().time())
    print('Batch resave done at ', datetime.now().time())

    return im

def useCourseMC(im):
    if not os.path.exists(os.path.join(file_path,'imageFrames_coarse_mc.tif')):
        print('Running coarse motion correction...')
        im = tfl.imread(outfile_stim_bg)
        frames_to_align_to = r_[:1000]
    
        im_align = ptMH.image.align_stack(im,frames_to_align_to,do_plot=True)
        tfl.imwrite(os.path.join(file_path,'imageFrames_coarse_mc.tif'),im_align,bigtiff=True)
        print('Coarse motion correction done...\n')
        file_name = 'imageFrames_coarse_mc.tif'

    return im_align
    