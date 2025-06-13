import numpy as np
import tifffile
import os
from skimage import transform

r_ = np.r_

def SI_batch_resave(infile, outfile, nFrChunk=300, rewriteOk=False, downscaleTuple=None, doPrint=True):
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
    with tifffile.TiffFile(infile) as tif:
        T = len(tif.pages)
        nR,nC = tif.pages[0].shape
    if downscaleTuple is not None:
        nR = int(nR/downscaleTuple[1])
        nC = int(nC/downscaleTuple[2])
    
    im = np.zeros((T,nR,nC))
    for fr in r_[0:T:nFrChunk]:
        if fr+nFrChunk <= T:
            ix = r_[fr:fr+nFrChunk]
            chunk = tifffile.imread(infile,key=ix)
            if doPrint:
                print(fr+nFrChunk, end=' ')
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')
        if fr+nFrChunk > T:
            ix = r_[fr:T]
            chunk = tifffile.imread(infile, key=ix)
            if doPrint:
                print(T)
            if downscaleTuple is not None:
                chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            im[ix,:,:] = chunk.astype('int16')

    im = im.astype('int16')
    tifffile.imsave(outfile, im, bigtiff=True)
    if doPrint:
        print('done. saved to {}'.format(outfile))
    return im

def makePositive(im):
    if np.min(im) < 0:
        im += abs(np.min(im))
    return im
