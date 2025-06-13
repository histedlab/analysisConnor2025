import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import tifffile as io
import scipy.signal as sig
import csv

def crossValSeq(dataMat,**kwargs):
    # if options dictionary not provided, set options to reasonable? defaults
    optionsDict = kwargs.get('optionsDict',{'testProp':.5,'alpha':0.05,'propRepThresh':0.05})
    # get data shape
    numTrials = dataMat.shape[1]
    numTimepoints = dataMat.shape[2]
    
    # calculate number of trials for test and holdout sets
    numTest = np.int(np.ceil(numTrials*optionsDict['testProp']))
    numHoldout = numTrials-numTest
    
    # Produce permuatation of trials
    sel = np.random.choice(np.arange(0,numTrials),numTrials,replace=False)
    
    # split data into sets
    testSet = dataMat[:,sel[0:numTest],:]
    holdoutSet = dataMat[:,sel[numTest:],:]
    
    # calculate frameswise difference across all trials
    testSetDiff = np.diff(testSet,axis=2)
    holdoutSetDiff = np.diff(holdoutSet,axis=2)
    
    # statistically test if difference for each cell at each timepoint is sig at alpha
    tTestSet,pTestSet = stats.ttest_1samp(testSetDiff,popmean=0,axis=1)
    tHoldoutSet,pHoldoutSet = stats.ttest_1samp(holdoutSetDiff,popmean=0,axis=1)
    
    # threshold based on positive difference beyond alpha
    statisticalThreshTest = (tTestSet>0) & (pTestSet < optionsDict['alpha'])
    statisticalThreshHoldout = (tHoldoutSet>0) & (pHoldoutSet < optionsDict['alpha'])
    
    # collect cell indicies from test set
    cellInd = []
    for t in np.arange(0,statisticalThreshTest.shape[1]):
        cellInd.append(np.where(statisticalThreshTest[:,t])[0])
    
    # num cells crossing alpha in test set
    numCells = np.sum(statisticalThreshTest,axis=0)
    
    # num of cells crossing test in 
    propRep = np.sum(statisticalThreshTest & statisticalThreshHoldout,axis=0)/numCells
    numConn = np.zeros(propRep.shape)
    for t in np.arange(0,propRep.shape[0]):
        count = 0
        currentVal = propRep[t,] > optionsDict['propRepThresh']
        while currentVal > 0:
            currentVal = propRep[t+count] > optionsDict['propRepThresh']
            count = count+1
        numConn[t,] = count
    print('There are '+np.str(np.sum(numConn>=3))+' timepoints connected to at least 3 conseq. frames that exceed the replication threshold')
    print('Max connected timepoints: '+np.str(np.amax(numConn)))
    outputDict = {'numCells':numCells,
                   'propRep':propRep,
                   'numConn':numConn,
                   'cellInd':cellInd,
                   'testTrials':sel[0:numTest],
                   'holdoutTrials':sel[numTest:],
    }
    return outputDict

def generateCellSelection(cellMasks,meanImage,cellInd,frameNum,outputDir,**kwargs):
    
    FOVPix = kwargs.get('FOVPix',512)
    
    cellInd = cellInd[frameNum]
    
    overlayImage = np.zeros((cellMasks.shape[1],cellMasks.shape[2],3))
    
    overlayImage[:,:,0] = meanImage/np.amax(meanImage.ravel())
    overlayImage[:,:,1] = meanImage/np.amax(meanImage.ravel())
    overlayImage[:,:,2] = meanImage/np.amax(meanImage.ravel())
    
    scalar = FOVPix/cellMasks.shape[1]
    cellCoords = np.zeros((cellInd.shape[0],2))
    
    count = 0
    for cells in cellInd:
        overlayImage[:,:,0] = overlayImage[:,:,0] + (cellMasks[cells,:,:]>0)
        cellCoords[count,:] = np.median(np.where(cellMasks[cells,:,:]),1)*scalar
        count = count+1
    io.imsave(os.path.join(outputDir,'cellSelection'+np.str(frameNum)+'.tif'),overlayImage)

    header = ['index', 'X', 'Y']
    full_data = np.zeros((cellInd.shape[0]+1,3))
    full_data[:,0] = np.arange(0,cellInd.shape[0]+1)
    full_data[0,1] = np.median(cellCoords,0)[1]
    full_data[0,2] = np.median(cellCoords,0)[0]
    full_data[1:,1] = cellCoords[:,1]
    full_data[1:,2] = cellCoords[:,0]
    
    with open(os.path.join(outputDir,'cellCoords'+np.str(frameNum)+'.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(full_data)
    return cellCoords

def generateOddEvenMovie(dataMat,cellMasks,outputFolder):
    trialLength = dataMat.shape[2]
    numTrials = dataMat.shape[1]
    numCells = dataMat.shape[0]
    
    FOVSize = cellMasks.shape[1]
    
    meanFull = np.mean(dataMat,1) 
    meanEven = np.mean(dataMat[:,0::2],1)
    meanOdd = np.mean(dataMat[:,1::2,:],1)
    
    deconvImageFull = np.zeros((trialLength,FOVSize,FOVSize))
    deconvImageEven = np.zeros((trialLength,FOVSize,FOVSize))
    deconvImageOdd = np.zeros((trialLength,FOVSize,FOVSize))

    
    # for each timepoint, loop through each cell and add cell mask scaled by deconv activity
    for time in np.arange(0,trialLength):
        for cells in np.arange(0,numCells):
            deconvImageFull[time,:,:] = deconvImageFull[time,:,:] + (meanFull[cells,time] * (cellMasks[cells,:,:]>0))
            deconvImageEven[time,:,:] = deconvImageEven[time,:,:] + (meanEven[cells,time] * (cellMasks[cells,:,:]>0))
            deconvImageOdd[time,:,:] = deconvImageOdd[time,:,:] + (meanOdd[cells,time] * (cellMasks[cells,:,:]>0))
            
    stitchedImage = np.ones((trialLength,FOVSize,FOVSize*3+2))

    stitchedImage[:,:,0:FOVSize] = deconvImageFull
    stitchedImage[:,:,FOVSize+1:FOVSize*2+1] = deconvImageEven
    stitchedImage[:,:,FOVSize*2+2:FOVSize*3+2] = deconvImageOdd


    io.imsave(os.path.join(outputFolder,'deconvFullEvenOdds.tiff'),stitchedImage)
    
    return

def generateIntersectionMatrix(dataMat,alpha,sizeSeq):
    # calculate frameswise difference across all trials
    dataMatDiff = np.diff(dataMat,axis=2)
    
    # statistically test if difference for each cell at each timepoint is sig at alpha
    tDiff,pDiff = stats.ttest_1samp(dataMatDiff,popmean=0,axis=1)
    
    sigDelta = pDiff<alpha
    
    intersectMat = np.zeros((dataMatDiff.shape[2],dataMatDiff.shape[2]))
    intersectMatNorm = np.zeros((dataMatDiff.shape[2],dataMatDiff.shape[2]))
    for t1 in np.arange(0,dataMatDiff.shape[2]):
        for t2 in np.arange(0,dataMatDiff.shape[2]):
            intersectMat[t1,t2] = np.sum(sigDelta[:,t1]*sigDelta[:,t2])
            intersectMatNorm[t1,t2] = intersectMat[t1,t2]/(np.amin([np.sum(sigDelta[:,t1]),np.sum(sigDelta[:,t2])])+1)
            
    intersectMatNormDiags = sig.correlate2d(intersectMatNorm,np.eye(sizeSeq),mode='same')
    
    plt.figure(figsize=[30,30])
    plt.rcParams.update({'font.size': 22})
    
    plt.subplot(1,3,1)
    plt.title('Intersection Matrix')
    plt.ylabel('Frame')
    plt.imshow(intersectMat)
    plt.subplot(1,3,2)
    plt.title('Norm. Intersection Matrix')
    plt.xlabel('Frame')
    plt.imshow(intersectMatNorm,vmin=0,vmax=.5)
    plt.subplot(1,3,3)
    plt.title('Filt. Norm. Intersection Matrix')
    plt.imshow(intersectMatNormDiags,vmin=.2,vmax=0.8)
    
    return intersectMatNorm
