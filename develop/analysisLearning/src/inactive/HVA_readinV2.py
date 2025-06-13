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
def mwmm2 (df): 

    df.loc[(df['Animal'] == 1707) & (df['Area'] == 'V1') & (df['Date'] <= 190201), 'brainPower'] = 3.06
    df.loc[(df['Animal'] == 1707) & (df['Area'] == 'V1') & (df['Date'] >= 190301), 'brainPower'] = 1.80 
    
    df.loc[(df['Animal'] == 1712) & (df['Area'] == 'V1') & (df['Date'] <= 190703), 'brainPower'] = 2.02
    df.loc[(df['Animal'] == 1712) & (df['Area'] == 'V1') & (df['Date'] >= 190703), 'brainPower'] = 1.31
    df.loc[(df['Animal'] == 1712) & (df['Area'] == 'PM'), 'brainPower'] = 2.49

    df.loc[(df['Animal'] == 1729) & (df['Area'] == 'V1'), 'brainPower'] = 1.96
    df.loc[(df['Animal'] == 1729) & (df['Area'] == 'PM'), 'brainPower'] = 3.0
    df.loc[(df['Animal'] == 1729) & (df['Area'] == 'control'), 'brainPower'] = 3.53
    
    df.loc[(df['Animal'] == 1730) & (df['Area'] == 'V1') & (df['Date'] <= 190703), 'brainPower'] = 2.59
    df.loc[(df['Animal'] == 1730) & (df['Area'] == 'V1') & (df['Date'] >= 190703), 'brainPower'] = 0.98
    df.loc[(df['Animal'] == 1730) & (df['Area'] == 'PM'), 'brainPower'] = 1.97
   
    df.loc[(df['Animal'] == 2009) & (df['Area'] == 'RL') & (df['Date'] <= 190902), 'brainPower'] = 0.36
    df.loc[(df['Animal'] == 2009) & (df['Area'] == 'RL') & (df['Date'] >= 190903), 'brainPower'] = 0.83
    df.loc[(df['Animal'] == 2009) & (df['Area'] == 'LM'), 'brainPower'] = 1.39
    df.loc[(df['Animal'] == 2009) & (df['Area'] == 'control'), 'brainPower'] = 2.73
    df.loc[(df['Animal'] == 2009) & (df['Area'] == 'PM'), 'brainPower'] = 0.87
    
    df.loc[(df['Animal'] == 2012) & (df['Area'] == 'LM') & (df['Date'] <= 200112), 'brainPower'] = 1.02
    df.loc[(df['Animal'] == 2012) & (df['Area'] == 'LM') & (df['Date'] >= 200112), 'brainPower'] = 1.22
    
    df.loc[(df['Animal'] == 2014) & (df['Area'] == 'V1'), 'brainPower'] = 0.49
  
    df.loc[(df['Animal'] == 2224) & (df['Area'] == 'PM'), 'brainPower'] = 0.44

    df.loc[(df['Animal'] == 2226) & (df['Area'] == 'PM') & (df['Date'] <= 191120), 'brainPower'] = 1.05
    df.loc[(df['Animal'] == 2226) & (df['Area'] == 'PM') & (df['Date'] >= 191121), 'brainPower'] = 1.21
    df.loc[(df['Animal'] == 2226) & (df['Area'] == 'V1'), 'brainPower'] = 1.27
    df.loc[(df['Animal'] == 2226) & (df['Area'] == 'AL'), 'brainPower'] = 1.19
    
    df.loc[(df['Animal'] == 2311) & (df['Area'] == 'RL'), 'brainPower'] = 1.0
    df.loc[(df['Animal'] == 2311) & (df['Area'] == 'PM'), 'brainPower'] = 1.81
    df.loc[(df['Animal'] == 2311) & (df['Area'] == 'V1'), 'brainPower'] = 0.97
    df.loc[(df['Animal'] == 2311) & (df['Area'] == 'LM'), 'brainPower'] = 1.12
   
    df.loc[(df['Animal'] == 2315) & (df['Area'] == 'PM') & (df['Date'] <= 200112), 'brainPower'] = 1.38
    df.loc[(df['Animal'] == 2315) & (df['Area'] == 'PM') & (df['Date'] >= 200112) & (df['Date'] <= 200123), 'brainPower'] = 1.23
    df.loc[(df['Animal'] == 2315) & (df['Area'] == 'PM') & (df['Date'] >= 200123), 'brainPower'] = 1.38
    
    df.loc[(df['Animal'] == 1981) & (df['Date'] == 190529), 'brainPower'] = 14.26 
    df.loc[(df['Animal'] == 1981) & (df['Date'] == 190530), 'brainPower'] = 20 
    
    df.loc[(df['Animal'] == 2664) & (df['Area'] == 'LM'), 'brainPower'] = 0.85
    
    df.loc[(df['Animal'] == 2665) & (df['Area'] == 'V1'), 'brainPower'] = 0.53
    df.loc[(df['Animal'] == 2665) & (df['Area'] == 'control'), 'brainPower'] = 2.67
 
    #discluded spots 
    #i2012.loc[i2012['Area'] == 'LM', 'brainPower'] = 0.63 -- over metabond
    #i2014.loc[i2014['Area'] == 'LM', 'brainPower'] = 0.29 -- blood spots
    #i1983.loc[i1983['Area'] == 'LM', 'brainPower'] = 0.5 -- blood spots
    #i1983.loc[i1983['Area'] == 'PM', 'brainPower'] = 0.4 -- blood spots
    #i1707.loc[i1707['Area'] == 'PM', 'brainPower'] = 1.68 -- not in PM / bad data 
    #i1707.loc[i1707['Area'] == 'LM', 'brainPower'] = 1.68 -- over metabond
    
    return (df)

def HVAinclude (df, B1_upper, B2_upper, nLevels):
    a = np.array(df['minTrials'].values.tolist())

    df['minTrials'] = np.where(a > 160, 160, a).tolist()

    df['mWmm2'] = round(df['brainPower'] * df['Laser Power'], 2)

    ## Drops days with less than 80% correct at the top power level 
    droplapses = df[df['B1 upper'] >= B1_upper]
    droplapses = droplapses[droplapses['B2 upper'] >= B2_upper]

    ## Drops days with less than 160 corrects+fails
    droptoofew = droplapses[droplapses['n usable trials'] >= droplapses['minTrials'] ]
    
    # Drops days with less than 5 levels 
    dropMaxC = droptoofew[droptoofew['nLevels'] >= nLevels]

    #Drops the days where the animal just didn't want to play 
    dropweirdos = dropMaxC[dropMaxC['% Increase'] <= 500]


    return (dropweirdos)

