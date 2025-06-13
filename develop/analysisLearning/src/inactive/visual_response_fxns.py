import os
import matplotlib.pyplot as plt
import numpy as np
import pytoolsMH as ptMH

a_ = np.asarray

# ##### analysis fxns

def compute_respOri(respDir,directions):
    # fold and average direction data in half to represent response to stim orientations
    nOri = len(directions)//2
    respOri = np.stack((respDir[:,0:nOri],respDir[:,nOri:]),axis=2).mean(axis=2)

    return respOri


def compute_OSI(respOri,activation=None):
    '''to calculate the OSI'''
    if activation == 'pos':
        respOri[respOri<0] = 0
    elif activation == 'neg':
        respOri[respOri>0] = 0
        respOri = np.abs(respOri)
    else:
        respOri = np.abs(respOri)
    
    # calculate preferred orientation and orthogonal direction
    prefOri = np.argmax(respOri,axis=1)
    orthOri = prefOri + 2
    orthOri[np.where(orthOri>3)] -= 4
    # OSI calculation using prefOri and orthOri w/ linear interpolation
    R_prefOri = a_([respOri[i,d] for i,d in enumerate(prefOri)])
    R_orthOri = a_([respOri[i,d] for i,d in enumerate(orthOri)])
    OSI = (R_prefOri - R_orthOri)/(R_prefOri + R_orthOri)
    OSI[np.isnan(OSI)] = 0

    return prefOri,R_prefOri,orthOri,R_orthOri,OSI


def compute_gOSI(respDir,directions,activation=None):
    '''to calculate the circular gaussian OSI'''
    if activation == 'pos':
        respDir[respDir<0] = 0
    elif activation == 'neg':
        respDir[respDir>0] = 0
        respDir = np.abs(respDir)
    else:
        respDir = np.abs(respDir)
        
    # define directions and convert to radians
    dir_rads = a_(directions)*(np.pi/180)
    # number of orientations
    nDir = len(directions)
    # gOSI calculation
    a = np.sum(respDir * np.cos(2*dir_rads),axis=1)
    b = np.sum(respDir * np.sin(2*dir_rads),axis=1)
    gOSI = np.sqrt((b**2 + a**2))/np.sum(respDir,axis=1)
    gOSI[np.isnan(gOSI)] = 0 #needed??

    return gOSI


def compute_DSI(respDir,activation=None):
    '''to calculate the DSI'''
    if activation == 'pos':
        respDir[respDir<0] = 0
    elif activation == 'neg':
        respDir[respDir>0] = 0
        respDir = np.abs(respDir)
    else:
        respDir = np.abs(respDir)
    
    # calculate preferred direction and opposite direction
    prefDir = np.argmax(respDir,axis=1)
    oppoDir = prefDir + 4
    oppoDir[np.where(oppoDir>7)] -= 8
    # DSI calculation using prefDir w/ linear interpolation
    R_prefDir = a_([respDir[i,d] for i,d in enumerate(prefDir)])
    R_oppoDir = a_([respDir[i,d] for i,d in enumerate(oppoDir)])
    DSI = (R_prefDir - R_oppoDir)/(R_prefDir + R_oppoDir)
    DSI[np.isnan(DSI)] = 0

    return prefDir,R_prefDir,oppoDir,R_oppoDir,DSI


def resultant_vector_length_trace(respDir,angles,axis=0,angle_unit='degrees'):    
    if angle_unit == 'degrees':
        angles_radians = np.deg2rad(angles)
    elif angle_unit == 'radians':
        angles_radians = angles
    else:
        raise 'angle_unit argument must be "degrees" or "radians"'
        
    resultant_x = np.tensordot(respDir,np.cos(angles_radians),axes=((axis),(0)))
    resultant_y = np.tensordot(respDir,np.sin(angles_radians),axes=((axis),(0)))
    resultant_length = np.sqrt(resultant_x**2 + resultant_y**2)
        
    return resultant_length


def resultant_vector_length_im(ims,angles,axis=0,angle_unit='degrees'):
    assert len(ims.shape) == 3, 'Input image must 3 dimensional (number of ims, xDim, yDim)'
    assert ims.shape[axis] == len(angles), 'There must be an equal number of images as there are angles'
    assert len(np.asarray(angles).shape) == 1, 'Angles must correspond to number of images and thus be 1 dimensional'
    
    if angle_unit == 'degrees':
        angles_radians = np.deg2rad(angles)
    elif angle_unit == 'radians':
        angles_radians = angles
    else:
        raise 'angle_unit argument must be "degrees" or "radians"'
        
    resultant_x = np.tensordot(ims,np.cos(angles_radians),axes=((axis),(0)))
    resultant_y = np.tensordot(ims,np.sin(angles_radians),axes=((axis),(0)))
    resultant_length = np.sqrt(resultant_x**2 + resultant_y**2)
        
    return resultant_length


# ##### plotting functions
# TO-DO: move this into plotting_2p_fxns.py

def plot_trialavg_cell_traces(traces,expt_params,nCell,stim_iC=None,expt_type='',ylim=None,smooth_span=None,save_path=None):
    # plot trial average data in each case
    # grab trial parameters
    nPreFr = expt_params['nPreFr']
    nStimFr = expt_params['nStimFr']
    nFrTrial = expt_params['nFrTrial']
    holoFrStarts = expt_params['holoFrStarts']
    nHoloFr = expt_params['nHoloFr']
    
    # generate plot [video, grating, diff compare]
    fig,ax = plt.subplots(figsize=np.r_[8,4]*2, nrows=1, ncols=1, sharex=True)
    fs = 16
    lw = 0.3
    
    if stim_iC == None:
        for iC in range(nCell):            
            trace = traces[iC,:]
            if smooth_span != None:
                trace = ptMH.math.smooth_lowess(trace,span=smooth_span)
            ax.plot(trace,color='black',lw=lw,alpha=0.7)
    else:
        for iC in range(nCell):
            trace = traces[iC,:]
            if smooth_span != None:
                trace = ptMH.math.smooth_lowess(trace,span=smooth_span)
            if iC in stim_iC:
                ax.plot(trace,color='magenta',lw=1,alpha=0.7)
            else:
                ax.plot(trace,color='black',lw=lw,alpha=0.7)
        

    # plot marker for stim on range (if there is simultaneous stim)
    if holoFrStarts != None:
    # for holo
        for j,startFr in enumerate(holoFrStarts):
            if j == 0:
                ax.axvspan(startFr,startFr+nHoloFr,color='magenta',alpha=0.1,label='Holo on')
            else:
                ax.axvspan(startFr,startFr+nHoloFr,color='magenta',alpha=0.1)
    else:
    # for vis
        ax.axvspan(nPreFr,(nPreFr+nStimFr),color='grey',alpha=0.2,label='Vis on')
    
    # plot 0 line
    ax.axhline(0,color='black',ls='--')

    # format plot
    ax.set_xlim((0,traces.shape[-1]-1))
    ax.set_xticks(np.asarray([60,120,180,240]))
    ax.set_xticklabels([0,2,4,6])
    ax.set_xlabel('Time from Stimulus Onset (s)',fontsize=fs)
    
    if ylim != None:
        ax.set_ylim(ylim)
    ax.set_ylabel('% dF/F',fontsize=fs)
    
    ax.tick_params(axis='both',which='major',length=0,labelsize=fs)
    
    ax.legend(edgecolor='white',bbox_to_anchor=(1, 1),loc='upper left',fontsize=fs)
    if smooth_span != None:
        title_string = 'Trial average responses to '+expt_type+'; # cells = '+str(nCell)+'; Lowess smooth span = '+str(smooth_span)
    else:
        title_string = 'Trial average responses to '+expt_type+'; # cells = '+str(nCell)
    ax.set_title(title_string,fontsize=fs)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save_path != None:
        plt.savefig(os.path.join(save_path,'trialavg_cell_responses_'+expt_type+'.png'),bbox_inches='tight',dpi=150)
