import os,sys
from pathlib import Path
import tifffile as tfl
from PIL import Image

import numpy as np
import pandas as pd
import math
from glob import glob

import scipy.io
from sklearn import preprocessing
from skimage import transform
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore

import mworksbehavior as mwb
from mworksbehavior import mwk_io
from mworksbehavior import mwkfiles
from mworksbehavior import mat_io
import pytoolsMH as ptMH

import matplotlib.pyplot as plt
import matplotlib as mpl

import shutil
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(category=FutureWarning, action="ignore")
    import caiman as cm

from IPython.display import clear_output
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.path.append(Path('../src').resolve().as_posix())
r_ = np.r_
a_ = np.asarray

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
            print('Frame count: ',T, end='. ')
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        del chunk

    im = im.astype('int16')
    tfl.imsave(outfile, im, bigtiff=True)
    print('saved to ~{}'.format(outfile[25:]))

def load_mat(folder, date, animal):
    file_path_root = Path(f'~/Desktop/data/{date}-i{animal}/{folder}/').expanduser() #ngf replace datadir w/ file_path
    file_path = file_path_root
    files = file_path_root / f'data2-i{animal}-{date}.mat'
    
    print(files.absolute().as_posix())

    mbs = mat_io.matBehavFile(files.absolute().as_posix())
    behavDf = mbs.df

    # remove fakemouse trials
    behavDf = behavDf[behavDf['tFakeMouseReactMs'].isnull()] #this doesn't actually remove anything in this case
    
    return file_path_root, file_path, behavDf

def load_mwk_output(file_path_nopath, file_path_path,stim_range='all',vis=False):
    print(file_path_path)
    try:
        mwkfile = glob(file_path_nopath+'/*.mwk*')[0]
    except:
        raise OSError('\nNo .mwk file found at specified path. Make sure path is defined correctly.')
    mwkname = os.path.join(file_path_nopath,mwkfile)  
    # for fixing up corrupt files
    useStimRange = stim_range
    # get imaging constants
    dd2 = mwb.imaging.consts.DataDir2p(file_path_nopath)
    print(dd2)
    # generate the h5 file from mwk
    mwb.mwk_io.mwk_to_h5(
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
        
    # Extract Ori Data
    data = pd.DataFrame(mwf.df)
    oris = 360 - np.array(data['value'].loc[data['tagname']=='tStim1DirectionDeg'])
    dirs = np.unique(oris)
    
    return mwk, data, oris, dirs

def align_tifs(numTifs, datadir, file_prefix, behavDf):
    movie_count = numTifs
    #generate movie timestamps
    tifNames = []

    for trNum in range(0, movie_count):
        numLength = len(str(trNum))
        if numLength < 5:
            numPadZero = 5 - numLength

            fileEnd = '0'*numPadZero + str(trNum)
            fileName = file_prefix + fileEnd + '.tif'

            tifNames.append(fileName)

    startTimes = []
    for tif in tifNames:
        infile = os.path.join(datadir, tif)
        with tfl.TiffFile(infile) as movie:
            meta = movie.pages[0].description.strip().split('\n')
            stampline= [i for i in meta if i.startswith('frameTimestamps_sec')]
            time = float(stampline[0].split(' ')[2])
            startTimes.append(time)

    m_trlen = []
    for i in range(0, len(startTimes)-1):
        tr_len = startTimes[i+1] - startTimes[i]
        m_trlen.append(tr_len)

    d = {'tif': [i for i in range(0, len(startTimes)-1)], 'length': m_trlen}
    movie_df = pd.DataFrame(data=d)


    #generate trial timestamps
    b_trlen = []
    HoldStartTimes = behavDf['tRealHoldStartTimeUs'].tolist()
    for i in range(0, behavDf.shape[0]-1):
        tr_len = (HoldStartTimes[i+1] - HoldStartTimes[i])/1000000
        b_trlen.append(tr_len)

    d = {'trial': [i for i in range(0, behavDf.shape[0]-1)], 'length': b_trlen}
    trial_df = pd.DataFrame(data=d)

    tifNames = tifNames[0:trial_df.shape[0]]

    #merge and compare timestamps
    mergedf = movie_df.merge(trial_df, left_on='tif', right_on='trial', how = 'inner')

    removeTifs = []
    y = 0
    i = 0 
    while i < mergedf.shape[0]:
        if round(mergedf.length_x[i]) != round(mergedf.length_y[y]):
            removeTifs.append(mergedf.tif[i])
            # print(i, y)
            i += 1

        elif round(mergedf.length_x[i]) == round(mergedf.length_y[y]):
            i += 1
            y += 1

    for tif in removeTifs:
        numLength = len(str(tif))
        if numLength < 5:
            numPadZero = 5 - numLength
            fileEnd = '0'*numPadZero + str(tif)
            fileName = file_prefix + fileEnd + '.tif'
        tifNames.remove(fileName)

        startTimes = []
    for tif in tifNames:
        infile = os.path.join(datadir, tif)
        with tfl.TiffFile(infile) as movie:
            meta = movie.pages[0].description.strip().split('\n')
            stampline = [i for i in meta if i.startswith('frameTimestamps_sec')]
            time = float(stampline[0].split(' ')[2])
            startTimes.append(time)

    # Redo mergedf with removed tifs
    m_trlen = []
    for i in range(0, len(startTimes)-1):
        tr_len = startTimes[i+1] - startTimes[i]
        m_trlen.append(tr_len)

    d = {'tif': [i for i in range(0, len(startTimes)-1)], 'length': m_trlen}
    movie_df = pd.DataFrame(data=d)
    mergedf = movie_df.merge(trial_df, left_on='tif', right_on='trial', how = 'inner')
    mergedf.to_csv(os.path.join(datadir,'filealign.csv'))
    
    return mergedf, tifNames, removeTifs

def align_trials(behavDf, mwkValues):
    removedIdx = []
    removeVals = 0
    y = 0
    i = 0 
    while i < behavDf.tStimDirectionDeg.shape[0]:
        if round(mwkValues.values[i]) != round(behavDf.tStimDirectionDeg.values[y]):
            removeVals += 1
            removedIdx.append(i)
            i += 1

        elif round(mwkValues.values[i]) == round(behavDf.tStimDirectionDeg.values[y]):
            i += 1
            y += 1
    return removeVals

def mwk_val(mwk, variable):
    val = mwk.df['value'][mwk.df['tagname']==variable]
    val = val[val != 0]
    return val 