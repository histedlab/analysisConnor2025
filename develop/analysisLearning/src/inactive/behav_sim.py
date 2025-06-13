import numpy as np
import scipy.stats as ss
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import random
import os, sys
import imp
from argparse import Namespace

a_ = np.asarray
r_ = np.r_

def computeGeoMean(geoMean, randMax):
    """See below for return namespace"""
    p = 1.0/geoMean
    binSizeMs = 1
    xs = r_[0:randMax:binSizeMs]
    yPmf = ss.geom(p).pmf(xs)
    #yCdf = ss.geom(p).cdf(xs)


    yNorm = yPmf/np.sum(yPmf)
    yHaz = np.zeros(len(xs))
    for iP in range(len(xs)):
        yHaz[iP] = yNorm[iP]/np.sum(yNorm[iP:])
    yCdf = np.cumsum(yNorm)

    return Namespace(yCdf=yCdf, yNorm=yNorm, yPmf=yPmf, yHaz=yHaz, xs=xs, binSizeMs=binSizeMs)

def lever_sim(geoMean, fixedHold, maxHold, reactTimeMs, tooFast, reactMean, reactSD, hazardRate, nTrials, bins):
    '''Creates the Lever Release Plot based on task and mouse parameters
    Arguments: 
        geoMean: task parameter for geometric distribution
        geoMeanFA: sets the geo distribution for FAs, usually set to 600
        fixedHold: task parameter for set minimum hold time
        maxHold: task parameter for max possible hold time 
        reactTimeMs: max alotted time for a response before considered a "miss"
        tooFast: if a mouse responses faster than this set time it is a false alarm
        reactMean: mouse parameter for their average reaction time 
        reactSD: mouse parameter for spread of responses 
        hazardRate: mouse parameter, false alarm hazard rate
        nTrials: number of trials
        bins: number of bins for histogram 
        
    Returns:
        hits: list of lever release times for a hit trial
        FAs: list of lever release times for an FA trial 
        miss: list of lever release times for a miss trial
        plots the histogram'''
    # fig = plt.figure(figsize=r_[2,0.75]*10, dpi=100)
    # gs = mpl.gridspec.GridSpec(1, 2)
    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    geoMeanFA = 600
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Release Time Selection
    i = 0
    hits = []
    FAs = []
    miss = []
    while i < nTrials:

        # Random Selection, Hit or FA Trial
        trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

        if trialType[0] == 0:
        # Single Trial Lever Release Selection
            randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
            totalHold = fixedHold + np.array(randHolds)
            reactTime = np.random.normal(loc=reactMean, scale=reactSD)
            levRelease = totalHold + reactTime
            # if fixedHold <= levRelease[0] <= maxHold:
            if reactTime >= reactTimeMs:
                miss.append(levRelease[0])
            elif reactTime <= tooFast:
                FAs.append(levRelease[0])
            else:
                hits.append(levRelease[0])
            i += 1

        if trialType[0] == 1:
        # Single Trial FA Lever Release Selection
            randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
            FAs.append(randFA[0])
            i += 1
    # Plot
    # ax = fig.add_subplot(gs[0])
    counts, bins, _ = plt.hist([hits, FAs, miss], bins, color=['C0', 'C1', 'C1'], stacked=True)
    # plt.show()
    # ax = fig.add_subplot(gs[1])
    # plt.plot(bins[:20], (counts[1]-counts[0])/(nTrials))
    # plt.show()
    return hits, FAs, miss


def lever_sim_interact(geoMean, fixedHold, maxHold, reactTimeMs, tooFast, reactMean, reactSD, hazardRate, nTrials, bins):

    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    geoMeanFA = 600
    FAn = computeGeoMean(geoMeanFA, maxHold)

    # Release Time Selection
    i = 0
    hits = []
    FAs = []
    miss = []
    while i < nTrials:
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                if reactTime >= reactTimeMs:
                    miss.append(levRelease[0])
                elif reactTime <= tooFast:
                    FAs.append(levRelease[0])
                else:
                    hits.append(levRelease[0])

                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    plt.hist([hits, FAs, miss], bins, color=['C0', 'C1', 'C1'], stacked=True)
    plt.show()
    
def lever_sim_laser_interact(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LhazardRate, nTrials, bins):

    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserFA = []
    FAs = []
    while i < nTrials:
        laser = random.choices(['on', 'off'])
        if laser[0] == 'on':
            trialType = random.choices([0,1], weights=[1-LhazardRate, LhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserFA.append(LaserRespt[0])
                i += 1
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    plt.hist([hits, FAs, laserFA], bins, stacked=True)
    plt.show()
    
def lever_sim_laser(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LhazardRate, nTrials, bins):
    # fig = plt.figure(figsize=r_[2,0.75]*10, dpi=100)
    # gs = mpl.gridspec.GridSpec(1, 2)

    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserFA = []
    FAs = []
    while i < nTrials:
        laser = random.choices(['on', 'off'])
        if laser[0] == 'on':
            trialType = random.choices([0,1], weights=[1-LhazardRate, LhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserFA.append(LaserRespt[0])
                i += 1
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    # ax = fig.add_subplot(gs[0])
    counts, bins, _ = plt.hist([hits, FAs, laserFA], bins, stacked=True)
    # plt.show()
    
    # ax = fig.add_subplot(gs[1])
    # plt.plot(bins[:20], ((counts[2]-counts[0])/(counts[2])))
    plt.show()
    return hits, FAs, laserFA, counts

def fa_plot(trials, days, peakRate, slopeRate, choiceRate):
    geoMean = 1000
    geoMeanFA = 600 # keep this fixed here 
    fixedHold = 600
    maxHold = 3500
    reactMean = 315
    reactSD = 100
    hazardRate = 0.3
    nTrials = trials
    bins = 20
    sens1 = []
    sens2 = []
    choice = []
    control = []
    for i in range(days):
        Lhits, LFAs, laserFA, Lcounts, FArate_1 = lever_sim_laser_rate(geoMean, geoMeanFA, fixedHold, 
                                                        maxHold, reactMean, reactSD, hazardRate, 
                                                        LhazardRate=peakRate, nTrials=nTrials, bins=bins)
        sens1.append(FArate_1)

        Lhits, LFAs, laserFA, Lcounts, FArate_2 = lever_sim_laser_rate(geoMean, geoMeanFA, fixedHold, 
                                                        maxHold, reactMean, reactSD, hazardRate, 
                                                        LhazardRate=slopeRate, nTrials=nTrials, bins=bins)
        sens2.append(FArate_2)

        Lhits, LFAs, laserFA, Lcounts, FArate_choice = lever_sim_laser_rate(geoMean, geoMeanFA, fixedHold, 
                                                        maxHold, reactMean, reactSD, hazardRate, 
                                                        LhazardRate=choiceRate, nTrials=nTrials, bins=bins)
        choice.append(FArate_choice)

        hits, FAs, counts, FArate_control = lever_sim_rate(geoMean, geoMeanFA, fixedHold, maxHold, 
                                                           reactMean, reactSD, hazardRate, nTrials, bins)
        control.append(FArate_control)
        
    plt.errorbar(x=['peak', 'slope', 'choice', 'control'], y=[np.mean(sens1), np.mean(sens2), np.mean(choice), np.mean(control)], 
                 yerr=[np.std(sens1)/np.sqrt(len(sens1)), np.std(sens2)/np.sqrt(len(sens2)), 
                       np.std(choice)/np.sqrt(len(choice)), np.std(control)]/np.sqrt(len(control)), fmt='o')
    plt.title('False Alarm Rates by Population')
    plt.ylabel('False Alarm Rate')
    plt.xlabel('Neuron Population')
    plt.show()
    
def lever_sim_laser_rate(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LhazardRate, nTrials, bins):
    # fig = plt.figure(figsize=r_[2,0.75]*10, dpi=100)
    # gs = mpl.gridspec.GridSpec(1, 2)

    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserFA = []
    FAs = []
    while i < nTrials:
        laser = random.choices(['on', 'off'])
        if laser[0] == 'on':
            trialType = random.choices([0,1], weights=[1-LhazardRate, LhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserFA.append(LaserRespt[0])
                i += 1
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
    FArate = (len(FAs)+len(laserFA))/(len(FAs)+len(hits)+len(laserFA))
    return hits, FAs, laserFA, counts, FArate

def lever_sim_rate(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, nTrials, bins):
    '''Creates the Lever Release Plot based on task and mouse parameters
    Arguments: 
        geoMean: task parameter for geometric distribution
        geoMeanFA: sets the geo distribution for FAs, usually set to 600
        fixedHold: task parameter for set minimum hold time
        maxHold: task parameter for max possible hold time 
        reactMean: mouse parameter for their average reaction time 
        reactSD: mouse parameter for spread of responses 
        hazardRate: mouse parameter, false alarm hazard rate
        nTrials: number of trials
        bins: number of bins for histogram 
        
    Returns:
        hits: list of lever release times for a hit trial
        FAs: list of lever release times for an FA trial 
        plots the histogram'''
    # fig = plt.figure(figsize=r_[2,0.75]*10, dpi=100)
    # gs = mpl.gridspec.GridSpec(1, 2)
    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Release Time Selection
    i = 0
    hits = []
    FAs = []
    while i < nTrials:

        # Random Selection, Hit or FA Trial
        trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

        if trialType[0] == 0:
        # Single Trial Lever Release Selection
            randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
            totalHold = fixedHold + np.array(randHolds)
            reactTime = np.random.normal(loc=reactMean, scale=reactSD)
            levRelease = totalHold + reactTime
            # if fixedHold <= levRelease[0] <= maxHold:
            hits.append(levRelease[0])
            i += 1

        if trialType[0] == 1:
        # Single Trial FA Lever Release Selection
            randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
            FAs.append(randFA[0])
            i += 1

    FArate = len(FAs)/(len(FAs)+len(hits))
    return hits, FAs, counts, FArate

def lever_sim_lasers(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LPhazardRate,
                    LShazardRate, LChazardRate, nTrials, bins):
    
    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserPeak = []
    laserSlope = []
    laserChoice = []
    FAs = []
    selection = []
    while i < nTrials:
        laser = random.choices(['peak', 'slope', 'choice', 'off'])
        selection.append(laser[0])
        if laser[0] == 'peak':
            trialType = random.choices([0,1], weights=[1-LPhazardRate, LPhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserPeak.append(LaserRespt[0])
                i += 1
        
        if laser[0] == 'slope':
            trialType = random.choices([0,1], weights=[1-LShazardRate, LShazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserSlope.append(LaserRespt[0])
                i += 1
        if laser[0] == 'choice':
            trialType = random.choices([0,1], weights=[1-LChazardRate, LChazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserChoice.append(LaserRespt[0])
                i += 1
                
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    counts, bins, _ = plt.hist([hits, FAs, laserPeak, laserSlope, laserChoice], bins, stacked=True)

    plt.show()
    return hits, FAs, laserPeak, laserSlope, laserChoice, selection

def lever_sim_lasers_interact(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LPhazardRate,
                    LShazardRate, LChazardRate, nTrials, bins):
    
    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserPeak = []
    laserSlope = []
    laserChoice = []
    FAs = []
    selection = []
    while i < nTrials:
        laser = random.choices(['peak', 'slope', 'choice', 'off'])
        selection.append(laser[0])
        if laser[0] == 'peak':
            trialType = random.choices([0,1], weights=[1-LPhazardRate, LPhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserPeak.append(LaserRespt[0])
                i += 1
        
        if laser[0] == 'slope':
            trialType = random.choices([0,1], weights=[1-LShazardRate, LShazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserSlope.append(LaserRespt[0])
                i += 1
        if laser[0] == 'choice':
            trialType = random.choices([0,1], weights=[1-LChazardRate, LChazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserChoice.append(LaserRespt[0])
                i += 1
                
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    counts, bins, _ = plt.hist([hits, FAs, laserPeak, laserSlope, laserChoice], bins, stacked=True)

    plt.show()
    
def lever_sim_lasers_rate(geoMean, geoMeanFA, fixedHold, maxHold, reactMean, reactSD, hazardRate, LPhazardRate,
                    LShazardRate, LChazardRate, nTrials, bins):
    
    # Hit Trials Distribution 
    n = computeGeoMean(geoMean, maxHold)

    # FA Trials Distribution
    FAn = computeGeoMean(geoMeanFA, maxHold)
    
    # Laser Trials Distribution
    # Get the time the laser turns on, fixed duration for now
    Mn = computeGeoMean(215, 400)


    # Release Time Selection
    i = 0
    hits = []
    laserPeak = []
    laserSlope = []
    laserChoice = []
    FAs = []
    selection = []
    while i < nTrials:
        laser = random.choices(['peak', 'slope', 'choice', 'off'])
        selection.append(laser[0])
        if laser[0] == 'peak':
            trialType = random.choices([0,1], weights=[1-LPhazardRate, LPhazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserPeak.append(LaserRespt[0])
                i += 1
        
        if laser[0] == 'slope':
            trialType = random.choices([0,1], weights=[1-LShazardRate, LShazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserSlope.append(LaserRespt[0])
                i += 1
        if laser[0] == 'choice':
            trialType = random.choices([0,1], weights=[1-LChazardRate, LChazardRate])
        
            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1
            
            if trialType[0] == 1:
                Lt = np.random.uniform(100, 500)
                randHolds = random.choices(np.arange(0, 400), weights = Mn.yNorm)
                reactTime = np.random.normal(loc=100, scale=40)
                MRt = np.array(randHolds) + reactTime
                LaserRespt = Lt + MRt
                laserChoice.append(LaserRespt[0])
                i += 1
                
        if laser[0] == 'off':
            
            # Random Selection, Hit or FA Trial
            trialType = random.choices([0,1], weights=[1-hazardRate, hazardRate])

            if trialType[0] == 0:
            # Single Trial Lever Release Selection
                randHolds = random.choices(np.arange(0, maxHold), weights = n.yNorm)
                totalHold = fixedHold + np.array(randHolds)
                reactTime = np.random.normal(loc=reactMean, scale=reactSD)
                levRelease = totalHold + reactTime
                # if fixedHold <= levRelease[0] <= maxHold:
                hits.append(levRelease[0])
                i += 1

            if trialType[0] == 1:
            # Single Trial FA Lever Release Selection
                randFA = random.choices(np.arange(0, maxHold), weights = FAn.yNorm)
                FAs.append(randFA[0])
                i += 1
        # Plot
    # counts, bins, _ = plt.hist([hits, FAs, laserPeak, laserSlope, laserChoice], bins, stacked=True)
    vals, counts = np.unique(selection, return_counts = True)
    FAratePeak = len(laserPeak)/counts[vals=='peak']
    FArateSlope = len(laserSlope)/counts[vals=='slope']
    FArateChoice = len(laserChoice)/counts[vals=='choice']
    FArateControl = len(FAs)/counts[vals=='off']
    
    return hits, FArateControl, FAratePeak, FArateSlope, FArateChoice, selection

def fa_plot_lasers(trials, days, faRate, peakRate, slopeRate, choiceRate):
    geoMean = 1000
    geoMeanFA = 600 # keep this fixed here 
    fixedHold = 600
    maxHold = 3500
    reactMean = 315
    reactSD = 100
    hazardRate = faRate
    LPhazardRate = peakRate
    LShazardRate = slopeRate
    LChazardRate = choiceRate
    nTrials = 450
    bins = 20
    nTrials = trials
    bins = 20
    sens1 = []
    sens2 = []
    choice = []
    control = []
    for i in range(days):
        hits, FArateControl, FAratePeak, FArateSlope, FArateChoice, selection = lever_sim_lasers_rate(geoMean, geoMeanFA, fixedHold, maxHold, 
                                                                                    reactMean, reactSD,hazardRate, LPhazardRate, 
                                                                                    LShazardRate, LChazardRate, nTrials, bins)

        sens1.append(FAratePeak)
        sens2.append(FArateSlope)
        choice.append(FArateChoice)
        control.append(FArateControl)
        
    print(FArateControl)
    plt.errorbar(x=['peak', 'slope', 'choice', 'control'], y=[np.mean(sens1), np.mean(sens2), np.mean(choice), np.mean(control)], 
                 yerr=[np.std(sens1)/np.sqrt(len(sens1)), np.std(sens2)/np.sqrt(len(sens2)), 
                       np.std(choice)/np.sqrt(len(choice)), np.std(control)]/np.sqrt(len(control)), fmt='o')
    plt.title('False Alarm Rates by Population')
    plt.ylabel('False Alarm Rate')
    plt.xlabel('Neuron Population')
    plt.show()