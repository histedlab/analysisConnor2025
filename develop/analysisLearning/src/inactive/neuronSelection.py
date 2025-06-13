from sklearn import preprocessing
import numpy as np
import pandas as pd
import scipy.io
import os,sys
import math
import tifffile as tfl
from pathlib import Path
from skimage import transform
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
import mworksbehavior as mwb
from mworksbehavior import mwk_io
from mworksbehavior import mwkfiles
from mworksbehavior import mat_io
from PIL import Image

import pytoolsMH as ptMH

import matplotlib.pyplot as plt
import matplotlib as mpl

import shutil

import seaborn as sns
sns.set_style('ticks')
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(category=FutureWarning, action="ignore")
    import caiman as cm

sys.path.append(Path('../src').resolve().as_posix())

import utils_2p_hd as u2p
import analysis_1p_fxns as f1p
import analysis_2p_fxns as a2p
import visual_response_fxns as vrf
import mworksbehavior as mwb
from mworksbehavior import mwk_io
from mworksbehavior import mwkfiles
import analysis_1p_fxns as f1p
from functions import *

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

r_ = np.r_
a_ = np.asarray

### Neuron Selection Functions

def plot_ori_trials(datadir, data, dirs):
    fig = plt.figure(figsize=(40, 20))
    top10 = []
    top10Idx = []
    for i in range(len(dirs)):
        fig.add_subplot(3, 5, i+1)
        traces_levdf = pd.DataFrame(data[i].mean(axis=0)).sort_values(by=90, ascending=False)
        sns.heatmap(traces_levdf[:20], cmap="RdBu_r")
        plt.axvline(60, color='black', lw=3)
        plt.axvline(120, color='black', lw=3)
        plt.title('Orientation: '+str(dirs[i]), fontsize=20)
        plt.ylabel('Neuron', fontsize=18)
        top10.append(traces_levdf[:10])
        top10Idx.append(traces_levdf[:10].index)
    plt.savefig(datadir / 'OriHeatmaps.png')
    return top10, top10Idx

def plot_ori_traces(datadir, data, dirs, topIdx):
    fig = plt.figure(figsize=(40, 20))
    for i in range(len(dirs)):
        fig.add_subplot(3, 5, i+1)
        for x in topIdx[i]:
            plt.plot(data[i][:,x,:].mean(axis=0))
            plt.title("Orientation: "+str(dirs[i]), fontsize=20)
            plt.xlabel('frames', fontsize=18)
            plt.ylabel('df/f', fontsize=18)
            plt.legend(labels=topIdx[i], fontsize=18)
    plt.savefig(datadir / 'OriTraces.png')
    
    
def tuning_curves_df(data, dirs, topIdx, frames): 
    oriDict = {'Neuron': [], 'Preferred Ori': [], 'Max Slope Ori': [], 'X Values': [], 'Average Responses': [], 'SEMs': [], 'Slope': []}
    for x in range(len(dirs)):
        # print(x)
        allSlopes = []
        allXvals = []
        fig = plt.figure(figsize=(50, 20))
        for i in range(10):
            oriDict['Neuron'].append(topIdx[x][i])
            neuron = topIdx[x][i]
            oriDict['Preferred Ori'].append(x)
            fig.add_subplot(2, 5, i+1)
            # Find traces for a single neuron at each orientation
            oriTraces = []
            for ori in range(data.shape[0]):
                oriTrace = []
                for trace in range(data.shape[1]):
                    oriTrace.append(data[ori,:,topIdx[x][i],:][trace][frames[0]+frames[1]:frames[0]+frames[2]])
                oriTraces.append(np.array(oriTrace))

            # Calculate averages and sems
            oriTraces = np.array(oriTraces)
            oriTracesAvg = oriTraces.mean(axis=2)
            avgs = []
            sems = []
            for ori in range(len(dirs)):
                avg = oriTracesAvg[ori].mean(axis=0)
                avgs.append(avg)
                sem = oriTracesAvg[ori].std()/np.sqrt(oriTracesAvg[0].shape[0])
                sems.append(sem)
            avgs = np.array(avgs)
            sems = np.array(sems)
            oriDict['Average Responses'].append(avgs)
            oriDict['SEMs'].append(sems)
            # Plot the tuning curves for each neuron
            # generate plot grid

            xs = dirs
            oriDict['X Values'].append(xs)
            slopes = []
            xvals = []
            for i in range(0, len(xs)-1):
                slope = ((avgs[i+1]-avgs[i])/(xs[i+1]-xs[i]))
                slopes.append(abs(slope))
                xvals.append(i)
            oriDict['Max Slope Ori'].append(np.array(slopes).argmax())
            oriDict['Slope'].append(np.max(slopes))
            allSlopes.append(np.array(slopes))
            allXvals.append(np.array(xvals))

            plt.plot(xs.astype(float),avgs,color='blue')
            sems_add = avgs+sems
            sems_min = avgs-sems
            plt.fill_between(xs.astype(float),sems_add,sems_min,color='blue',alpha=0.2)

            plt.xticks(xs,labels=xs,fontsize=12)
            plt.xlabel('Orientation (Degrees)',fontsize=15)
            # plt.xscale('log')
            plt.yticks(fontsize=12)
            # plt.axvline(40, color='black', ls='--')
            plt.ylabel('Average % dF/F Response',fontsize=15)
            plt.title('Neuron '+str(neuron),fontsize=18)
        # plt.show()
    
    tuningDf = pd.DataFrame(oriDict)
    tuningDf['Preferred Ori'] = tuningDf['Preferred Ori'].replace(np.arange(len(dirs)), dirs)
    avgResps = []
    for index, row in tuningDf.iterrows():
        for i, ori in enumerate(dirs):
            if row[1]==ori:
                avgResps.append(row[4][i])
    tuningDf["Pref Avgs"] = np.array(avgResps)
    widths = []
    for i, row in tuningDf.iterrows():
        widths.append(abs(int(np.diff(row[3][np.argsort(row[4])[0:2]]))))
    tuningDf['Widths'] = widths
    
    slopesIvals = []
    for i in range(dirs.shape[0]):
        if i < dirs.shape[0]-1:
            slopesIvals.append(f'{dirs[i]}-{dirs[i+1]}')
    tuningDf['Max Slope Ori'] = tuningDf['Max Slope Ori'].replace(np.arange(len(dirs)-1), slopesIvals)
    
    return tuningDf, slopesIvals


def plot_peak_stats(data):    
    fig = plt.figure(figsize=(12,5))
    fig.add_subplot(1, 2, 1)
    sns.swarmplot(x='Preferred Ori', y='Pref Avgs', data=data)
    plt.ylabel('Average Df/f Response (%)')
    plt.xlabel('Orientation (degrees)')
    plt.title('Average Response During Stim at Preferred Orientation')

    fig.add_subplot(1, 2, 2)
    plt.scatter(data['Widths'], data['Pref Avgs'])
    plt.ylabel('Average Response to Preferred Ori (df/f %)')
    plt.xlabel('Tuning Width')
    plt.title('Df/f Response by Tuning Width')
    plt.show()
    

def plot_selectedNeurs(selectedNeurs):
    fig = plt.figure(figsize=(50, 80))
    for i in range(selectedNeurs.index.shape[0]):
        fig.add_subplot(8, 5, i+1)
        xs = selectedNeurs['X Values'].iloc[i]
        avgs = selectedNeurs['Average Responses'].iloc[i]
        sems = selectedNeurs['SEMs'].iloc[i]
        plt.plot(xs.astype(float), avgs, color='blue')
        # plt.xscale('log')
        plt.fill_between(xs.astype(float),avgs+sems,avgs-sems,color='blue',alpha=0.2)
        plt.title('Neuron ' +str(selectedNeurs.Neuron.iloc[i]), fontsize=25)
        
        
def plot_slope_stats(data, slopes, dirs):    
    fig = plt.figure(figsize=(20,5))
    fig.add_subplot(1, 3, 1)
    sns.swarmplot(x='Max Slope Ori', y='Slope', data=data)
    plt.xticks(ticks=np.arange(len(dirs)-1), labels=slopes)
    plt.xlabel('Slope Orientation Ranges')
    plt.title('Max Slope by Orientation')

    fig.add_subplot(1, 3, 2)
    plt.scatter(data['Slope'], data['Pref Avgs'])
    plt.ylabel('Average Response to Preferred Ori (df/f %)')
    plt.xlabel('Slope')
    plt.title('Df/f Response by Slope')
    
    fig.add_subplot(1, 3, 3)
    plt.scatter(data['Widths'], data['Slope'])
    plt.xlabel('Tuning Widths')
    plt.ylabel('Slope')
    plt.title('Df/f Response by Slope')
    plt.show()
    

def trial_heatmap(trials, resTime, i):
    tr_one = pd.DataFrame(trials[i])
    sort = tr_one.sort_values(by=(resTime[i]-1), ascending=False)
    sns.heatmap(sort.iloc[:20,resTime[i]-20:resTime[i]+10], cmap="RdBu_r")
    plt.axvline(20, color='black', lw=5)
    plt.title('False Alarm Trial Responses')
    plt.show()
    

def selectChoice(df, delta, freq, i):
    fig = plt.figure(figsize=[2.5, 2.5])
    x = np.linspace(0,40, num=40) 
    x = np.asarray([int(np.round(t*33, 0)) for t in np.nan_to_num(x)])
    x = x/1000
    x = x-x[30]
    plt.plot(x, df['Avg Trace'][(df['Delta'] > delta) & (df['Frequency'] > freq)].iloc[i], color='blue')
    plt.axvline(x[30], color='black')
    plt.ylabel('Average Df/f (%)')
    plt.xlabel('time (s)')
    plt.title(f'Neuron ' + str(df.Neuron[(df['Delta'] > delta) & (df['Frequency'] > freq)].iloc[i]))
    plt.show()
    
def load_mwk(animal, date, vis):
    datadir = Path('~/Data/'+f'{date}-i{animal}/').expanduser()
    if vis: 
        mwkPath = os.path.join(datadir, f'{date}-i{animal}-vis.mwk2')

        if not os.path.exists(datadir/ f'{date}-i{animal}-vis.h5'):
            mwk_io.mwk_to_h5(mwkPath, keep_system_vars=True)

        mwkF = os.path.join(datadir, f'{date}-i{animal}-vis.h5')

        mwk = mwkfiles.MWKFile(mwkF)
    else:
        mwkPath = os.path.join(datadir, f'{date}-i{animal}.mwk2')

        if not os.path.exists(datadir/ f'{date}-i{animal}.h5'):
            mwk_io.mwk_to_h5(mwkPath, keep_system_vars=True)

        mwkF = os.path.join(datadir, f'{date}-i{animal}.h5')

        mwk = mwkfiles.MWKFile(mwkF)
        
    # Extract Ori Data
    data = pd.DataFrame(mwk.df)
    oris = 360 - np.array(data['value'].loc[data['tagname']=='tStim1DirectionDeg'])
    dirs = np.unique(oris)
    
    return datadir, mwk, data, oris, dirs

def load_suite2p(datadir):
    folder = datadir / 'suite_tif/suite2p/plane0/'
    F_raw = np.load(os.path.join(folder,'F.npy'), allow_pickle=True)
    isCell = np.load(os.path.join(folder,'iscell.npy'), allow_pickle=True)
    ops = np.load(os.path.join(folder,'ops.npy'), allow_pickle=True)
    stat = np.load(os.path.join(folder,'stat.npy'), allow_pickle=True)

    # restrict fluo traces to cells
    F = F_raw[isCell[:,0]==1,:]
    return ops, stat, F, isCell

def calc_dff_reshape(F, im, expt_params_list):
    dfof, df, f, basef = generate_dfof_cell_traces(F, im, expt_params_list)

    F_trials = dfof.reshape(dfof.shape[0], expt_params_list['nTrial'], expt_params_list['nFrTrial'])
    traces = []
    for i in range(F_trials.shape[1]):
        traces.append(F_trials[:,i,:])
    traces = np.array(traces)
    return traces, dfof, df, f, basef

def sort_by_ori(traces, dirs, oris, expt_params_list):
    oris = oris[:expt_params_list['nTrial']]
    traces_lev = []
    for ori in dirs:
        # print(ori)
        idx = np.where(oris == ori)
        traces_lev.append(np.array(traces[idx][:19]))
        # print(traces[idx].shape)
    traces_lev = np.array(traces_lev)
    return traces_lev


def load_mat(animal, date):
    datadir = Path('~/Data/'+f'{date}-i{animal}/').expanduser()

    files = datadir / f'data2-i{animal}-{date}.mat'

    mbs = mat_io.matBehavFile(files.absolute().as_posix())

    behavDf = mbs.df

    # remove fakemouse trials
    behavDf = behavDf[behavDf['tFakeMouseReactMs'].isnull()] #this doesn't actually remove anything in this case
    
    return datadir, behavDf

def align_tifs(numTifs, datadir, behavDf):
    movie_count = numTifs
    #generate movie timestamps
    tifNames = []

    for trNum in range(0, movie_count):
        numLength = len(str(trNum))
        if numLength < 5:
            numPadZero = 5 - numLength

            fileEnd = '0'*numPadZero + str(trNum)
            fileName = '920nm-behav-2_' + fileEnd + '.tif'

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
            fileName = '920nm-behav-2_' + fileEnd + '.tif'
        tifNames.remove(fileName)

        startTimes = []
    for tif in tifNames:
        infile = os.path.join(datadir, tif)
        with tfl.TiffFile(infile) as movie:
            meta = movie.pages[0].description.strip().split('\n')
            stampline= [i for i in meta if i.startswith('frameTimestamps_sec')]
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

def preproc_outcome(datadir, data, outcome, base_shift, res_shift):
    tifNums = data[(data.trOutcome == outcome)].index.tolist()
    Df = data.iloc[tifNums].reset_index()
    stim_fr = Df.trHoldFrs - Df.trReactionTimeFrs
    base_fr = (stim_fr - base_shift).tolist()
    res_fr = (stim_fr + res_shift).tolist()
    stim_fr = stim_fr.tolist()

    Df['base_fr'] = base_fr 
    Df['res_fr'] = res_fr
    Df['stim_fr'] = stim_fr
    
    preproc_folders(datadir, Df, outcome)
    
    tifNames = Df.tifNames.tolist()
    frames = [0]
    for tif in tifNames:
        print(tif)
        shape = tfl.imread(datadir / tif).shape[0]
        frames.append(shape)
    frame_idx = np.cumsum(frames)
    
    return Df, tifNums, frame_idx

def preproc_folders(datadir, df, outcome):
    tifs = df.tifNames.tolist()
    for tif in tifs:
        orig_path = os.path.join(datadir, tif)
        out_path = datadir/f'{outcome}'/f'out_{tif}'
        im = f1p.SI_batch_resave(orig_path, out_path, nFrChunk=1000, downscaleTuple=(1,2,2), rewriteOk=True)