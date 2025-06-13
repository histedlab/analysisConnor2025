'''To-do: 
    1) Figure out the correct variables to return for each function / make more helper functions so you're not inputting and outputting quite so much to get the job done ''' 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytoolsMH as ptMH
import pandas as pd
import seaborn as sns
import os,sys
import scipy.io

from mworksbehavior import mat_io
from mworksbehavior import psychfun
from pytoolsMH import dataio

sys.path.append(os.path.realpath('../src'))


sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')

import HVA_readin as HVA 

def setup (powers): 
    #get rid of control spots (for now)
    powers = powers[powers['Area'] != 'control']
    #Seperate data based on area
    V1 = powers[powers['Area'] == 'V1']
    medial = powers[powers['Area'] == 'PM']
    
    lateral = powers[(powers['Area'] == 'LM') | (powers['Area'] == 'RL')]
    #seperate data based on retinotopy
    ret = powers[powers['Retinotopy'] == 'retinotopic']
    nonret = powers[powers['Retinotopy'] == 'non-retinotopic']
    
    #rename some stuff 
    ret['Spot Location'] = 'Retinotopic'
    nonretMedial = nonret[nonret['Area'] == 'PM'] 
    nonretMedial['Area'] = 'Medial'
    nonretLateral = nonret[nonret['Area'] == 'LM']
    nonretLateral['Area'] = "Lateral"
    nonret = pd.concat([nonretMedial, nonretLateral])
    non2 = pd.concat([nonretMedial, nonretLateral])
    non2['Area'] = 'Control'
    retV1 = ret[ret['Area'] == 'V1']
    retM = ret[ret['Area'] == 'PM']
    retM['Area'] = 'Medial'
    retL = ret[(ret['Area'] == 'LM') | (ret['Area'] == 'RL')| (ret['Area'] == 'AL') | (ret['Area'] == 'control')]
    retL['Area'] = 'Lateral'
    retino = ret
    ret = pd.concat([retV1, retM, retL, non2])
    
    return (ret, retino, nonret, medial, lateral, retV1)

def plotAll(ret, palette, trunc, savefig):
    lm = sns.lmplot('mWmm2', '% Increase', data = ret, hue = 'Area', palette = palette, aspect = 1.5, truncate = trunc, ci = 95, n_boot = 1000, legend = False) 
    plt.xscale('log')
    plt.xlim(0.01, 10)
    axes = lm.axes
    #axes[0,0].set_xlim(0.01, 4)
    plt.ylim(-50, 200)
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    plt.title('Effects on All Mapped Spots')
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    plt.legend(('V1', 'Medial', 'Lateral', 'No Effect'))
    #plt.legend(labels = ['V1', 'Medial', 'Lateral', 'Non-Retinotopic'])
    if savefig == True: 
        plt.savefig('allspots.pdf')
        plt.savefig('allspots.png')

def plotNon(nonret, palette, trunc, savefig):
    lm = sns.lmplot('mWmm2', '% Increase', data = nonret, hue = 'Area', palette = palette, aspect = 1.5, truncate = trunc, ci = 95, n_boot = 1000, legend = False)
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(-50, 200)
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    plt.title('Spots with No Effect')
    plt.legend(('Medial', 'Lateral'))
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    if savefig == True:
        plt.savefig('nonretinospots.pdf')
        plt.savefig('nonretinospots.png')

def plotLateral(lateral, palette, trunc, savefig):
    lm = sns.lmplot('mWmm2', '% Increase', data = lateral, hue = 'Retinotopy', palette = palette, aspect = 1.5, truncate = trunc, ci = 95, n_boot = 1000, legend = False)
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(-50, 200)
    plt.title('Effects on Lateral Areas')
    plt.legend(('No Effect', 'Effect'))
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    if savefig == True: 
        plt.savefig('lateralspots.pdf')
        plt.savefig('lateralspots.png')

def plotMedial(medial, palette, trunc, savefig):
    pm = sns.lmplot('mWmm2', '% Increase', data = medial, hue = 'Retinotopy', palette = palette, aspect = 1.5, truncate = trunc, ci = 95, n_boot = 1000, legend = False)
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(-50, 200)
    plt.title('Effects on Medial Areas')
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    plt.legend(('No Effect', 'Effect'))
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    if savefig == True: 
        plt.savefig('medialspots.pdf')
        plt.savefig('medialspots.png')

def plotV1(retV1, palette, trunc, savefig): 
    lm = sns.lmplot('mWmm2', '% Increase', data = retV1, hue = 'Retinotopy', palette = palette, aspect = 1.5, ci = 95, n_boot = 1000, legend = False, truncate = trunc)
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(-50, 200)
    plt.title('Effects on Primary Visual Cortex')
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    if savefig == True: 
        plt.savefig('V1spots.png')
        plt.savefig('V1spots.pdf')

def plotretino (ret, palette, trunc, savefig):
    lm = sns.lmplot('mWmm2', '% Increase', data = ret, hue = 'Area', palette = palette, aspect = 1.5, ci = 95, n_boot = 1000, legend = False, truncate = trunc)
    plt.xscale('log')
    plt.xlim(0.01, 10)
    plt.ylim(-50, 200)
    plt.title('Spots with Effects')
    plt.xlabel('Laser Intensity (mW/mm^2)')
    plt.ylabel( '% Threshold Increase')
    plt.xticks((0.01, 0.1, 1.0, 10.0), (0.01, 0.1, 1.0, 10.0))
    plt.yticks((0, 100, 200), (0, 100, 200))
    if savefig == True: 
        plt.savefig('retinospots.png')
        plt.savefig('retinospots.pdf')

        

# ## TIMING STUFF 

def masks(timing):
    V1 = timing[timing['Area'] == 'V1']
    V1 = V1[V1['Laser Power'] == 0.2]
    V1 = V1[V1['Mask'] != 30]
    posMask = V1[V1['Mask'] >= 0]
    negMask = V1[V1['Mask'] <= 0]
    return (V1, posMask, negMask)

def plotTiming(V1, savefig):
    sns.lineplot('Mask', '% Increase', data = V1, color = 'Black', ci = 99.5)
    sns.scatterplot('Mask', '% Increase', data = V1, color = 'Black', alpha = 0.5) 
    plt.title('Timing Effects, all power levels')
    if savefig == True: 
        plt.savefig('allTiming.png')
        plt.savefig('allTiming.pdf')

def plotPos(posMask, savefig): 
    sns.lineplot('Mask', '% Increase', data = posMask, color = 'Black', ci = 99.5)
    sns.scatterplot('Mask', '% Increase', data = posMask, color = 'Black', alpha = 0.5)
    plt.ylim(-50, 200) 
    plt.xlim(0, 160)
    plt.title('Positive mask latencies, 0.2mW')
    plt.xlabel('Mask latency (ms)')
    if savefig == True: 
        plt.savefig('posTiming.png')
        plt.savefig('posTiming.pdf')

def plotNeg(negMask, savefig): 
    sns.lineplot('Mask', '% Increase', data = negMask, color = 'Black', ci = 99.5)
    sns.scatterplot('Mask', '% Increase', data = negMask, color = 'Black', alpha = 0.5)  
    plt.ylim(-50, 200)
    plt.xlim(-150, -40)
    plt.title ('Negative mask latencies, 0.2mW')
    plt.xlabel('Mask latency (ms)')
    if savefig == True: 
        plt.savefig('negTiming.png')
        plt.savefig('negTiming.pdf')

def forStats(powers):
    #get rid of control spots (for now)
    powers = powers[powers['Area'] != 'control']
    #Seperate data based on area
    V1 = powers[powers['Area'] == 'V1']
    medial = powers[powers['Area'] == 'PM']
    
    lateral = powers[(powers['Area'] == 'LM') | (powers['Area'] == 'RL')]
    #seperate data based on retinotopy
    ret = powers[powers['Retinotopy'] == 'retinotopic']
    nonret = powers[powers['Retinotopy'] == 'non-retinotopic']
    
    #rename some stuff 
    ret['Spot Location'] = 'Retinotopic'
    nonretMedial = nonret[nonret['Area'] == 'PM'] 
    nonretMedial['Area'] = 'Medial'
    nonretLateral = nonret[nonret['Area'] == 'LM']
    nonretLateral['Area'] = "Lateral"
    nonret = pd.concat([nonretMedial, nonretLateral])
    non2 = pd.concat([nonretMedial, nonretLateral])
    non2['Area'] = 'Control'
    retV1 = ret[ret['Area'] == 'V1']
    retM = ret[ret['Area'] == 'PM']
    retM['Area'] = 'Medial'
    retL = ret[(ret['Area'] == 'LM') | (ret['Area'] == 'RL')]
    retL['Area'] = 'Lateral'
    retino = ret
    ret = pd.concat([retV1, retM, retL, non2])
    
    return (nonretMedial, nonretLateral, retL, retM)

 
