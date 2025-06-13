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

# does the laser power -> laser power intensity conversions using the values we caluclate from spot size
def readin (path, earlies): 
    path = path
    #reads in each animals saved data 
    i1707 = pd.read_excel(path+'1707offset.xlsx') 
    i1707.loc[(i1707['Area'] == 'V1') & (i1707['Date'] <= 190201), 'brainPower'] = 3.06
    i1707.loc[(i1707['Area'] == 'V1') & (i1707['Date'] >= 190301), 'brainPower'] = 1.80
    #i1707.loc[i1707['Area'] == 'PM', 'brainPower'] = 1.68 -- not in PM 
    #i1707.loc[i1707['Area'] == 'LM', 'brainPower'] = 1.68 -- over metabond 

    i1712 = pd.read_excel(path+'1712offset.xlsx')
    i1712.loc[(i1712['Area'] == 'V1') & (i1712['Date'] <= 190703), 'brainPower'] = 2.02
    i1712.loc[(i1712['Area'] == 'V1') & (i1712['Date'] >= 190703), 'brainPower'] = 1.31
    i1712.loc[i1712['Area'] == 'PM', 'brainPower'] = 2.49

    i1729 = pd.read_excel(path+'1729offset.xlsx')
    i1729.loc[i1729['Area'] == 'V1', 'brainPower'] = 1.96
    i1729.loc[i1729['Area'] == 'PM', 'brainPower'] = 3.0
    i1729.loc[i1729['Area'] == 'control', 'brainPower'] = 3.53

    i1730 = pd.read_excel(path+'1730offset.xlsx')
    i1730.loc[(i1730['Area'] == 'V1') & (i1730['Date'] <= 190703), 'brainPower'] = 2.59
    i1730.loc[(i1730['Area'] == 'V1') & (i1730['Date'] >= 190703), 'brainPower'] = 0.98
    i1730.loc[i1730['Area'] == 'PM', 'brainPower'] = 1.97
    
    #i1983 = pd.read_excel(path+'1983offset.xlsx')
    #i1983.loc[i1983['Area'] == 'LM', 'brainPower'] = 0.5 -- blood spots
    #i1983.loc[i1983['Area'] == 'PM', 'brainPower'] = 0.4 -- blood spots

    i2009 = pd.read_excel(path+'2009offset.xlsx')
    i2009.loc[(i2009['Area'] == 'RL') & (i2009['Date'] <= 190902), 'brainPower'] = 0.36
    i2009.loc[(i2009['Area'] == 'RL') & (i2009['Date'] >= 190903), 'brainPower'] = 0.83
    i2009.loc[i2009['Area'] == 'LM', 'brainPower'] = 1.39
    i2009.loc[i2009['Area'] == 'control', 'brainPower'] = 2.73
    i2009.loc[i2009['Area'] == 'PM', 'brainPower'] = 0.87
    
    i2012 = pd.read_excel(path+'2012offset.xlsx')
    i2012.loc[(i2012['Area'] == 'LM') & (i2012['Date'] <= 200112), 'brainPower'] = 1.02
    i2012.loc[(i2012['Area'] == 'LM') & (i2012['Date'] >= 200112), 'brainPower'] = 1.22
    #i2012.loc[i2012['Area'] == 'LM', 'brainPower'] = 0.63 -- over metabond

    i2014 = pd.read_excel(path+'2014offset.xlsx')
    #i2014.loc[i2014['Area'] == 'LM', 'brainPower'] = 0.29 -- blood spots
    i2014.loc[i2014['Area'] == 'V1', 'brainPower'] = 0.49

    i2224 = pd.read_excel(path+'2224offset.xlsx')
    i2224.loc[i2224['Area'] == 'PM', 'brainPower'] = 0.44
    
    i2226 = pd.read_excel(path+'2226offset.xlsx')
    i2226.loc[(i2226['Area'] == 'PM') & (i2226['Date'] <= 191120), 'brainPower'] = 1.05
    i2226.loc[(i2226['Area'] == 'PM') & (i2226['Date'] >= 191121), 'brainPower'] = 1.21
    i2226.loc[i2226['Area'] == 'V1', 'brainPower'] = 1.27
    i2226.loc[i2226['Area'] == 'AL', 'brainPower'] = 1.19
    
    i2311 = pd.read_excel(path+'2311offset.xlsx')
    i2311.loc[i2311['Area'] == 'RL', 'brainPower'] = 1.0
    i2311.loc[i2311['Area'] == 'PM', 'brainPower'] = 1.81
    i2311.loc[i2311['Area'] == 'V1', 'brainPower'] = 0.97
    i2311.loc[i2311['Area'] == 'LM', 'brainPower'] = 1.12
    
    i2315 = pd.read_excel(path+'2315offset.xlsx')
    i2315.loc[(i2315['Area'] == 'PM') & (i2315['Date'] <= 200112), 'brainPower'] = 1.38
    i2315.loc[(i2315['Area'] == 'PM') & (i2315['Date'] >= 200112) & (i2315['Date'] <= 200123), 'brainPower'] = 1.23
    i2315.loc[(i2315['Area'] == 'PM') & (i2315['Date'] >= 200123), 'brainPower'] = 1.38
    
    i1981 = pd.read_excel(path+'1981offset.xlsx')
    
    readin = pd.concat([i1712, i1730, i1707, i1729, i2012, i2224, i2014, i2009, i2311, i2226, i2315, i1981]) 
    
    readin['Mask'] = -1*((readin['Offset'])+45)
    
    if earlies == False: 
        readin = readin.drop(labels = ['B1 earlies', 'B2 earlies'], axis = 1)
    else: 
        readin = readin
    
    return (readin)

def HVAinclude (readin, B1_upper, B2_upper, nLevels, offset):
    a = np.array(readin['minTrials'].values.tolist())

    readin['minTrials'] = np.where(a > 160, 160, a).tolist()

    readin['mWmm2'] = round(readin['brainPower'] * readin['Laser Power'], 2)

    ## Drops days with less than 80% correct at the top power level 
    droplapses = readin[readin['B1 upper'] >= B1_upper]
    droplapses = droplapses[droplapses['B2 upper'] >= B2_upper]

    ## Drops days with less than 160 corrects+fails
    droptoofew = droplapses[droplapses['n usable trials'] >= droplapses['minTrials'] ]
    
    # Drops days with less than 5 levels 
    dropMaxC = droptoofew[droptoofew['nLevels'] >= nLevels]

    #Drops the days where the animal just didn't want to play 
    dropweirdos = dropMaxC[dropMaxC['% Diff'] <= 500]

    # Cuts it down to -100ms offsets 
    timing = dropweirdos[dropweirdos['Offset'] == offset]

    return (timing)

def timingInclude (readin, B1_upper, B2_upper, nLevels, powers): 
    a = np.array(readin['minTrials'].values.tolist())

    readin['minTrials'] = np.where(a > 160, 160, a).tolist()

    readin['mWmm2'] = round(readin['brainPower'] * readin['Laser Power'], 2)
  
    readin['B1 upper'] = round(2)
    readin['B2 upper'] = round(2)

    ## Drops days with less than 80% correct at the top power level 
    droplapses = readin[readin['B1 upper'] >= B1_upper]
    droplapses = droplapses[droplapses['B2 upper'] >= B2_upper]

    ## Drops days with less than 160 corrects+fails
    droptoofew = droplapses[droplapses['n usable trials'] >= droplapses['minTrials'] ]

    dropMaxC = droptoofew[droptoofew['nLevels'] >= nLevels]

    ##Drops the days where the animal just didn't want to play 
    timing = dropMaxC[dropMaxC['% Diff'] <= 400]
    

    #Pulls out the power levels that we're comparing at 
    if powers == True: 
        powers = timing[(timing['Laser Power'] == 0.05) | (timing['Laser Power'] == 0.10) | (timing['Laser Power'] == 0.1)
        |(timing['Laser Power'] == 0.2) | (timing['Laser Power'] == 0.3) | (timing['Laser Power'] == 0.4) 
        |(timing['Laser Power'] == 0.8) | (timing['Laser Power'] == 1.2) |(timing['Laser Power'] == 1.6)] 

    powers = powers.dropna()
    powers.rename(columns = {'retino':'Retinotopy'}, inplace = True)
    return (timing)

