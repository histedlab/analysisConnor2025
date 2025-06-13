import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage import transform
import tifffile as tfl

# import custom packages
import caiman as cm
import pytoolsMH as ptMH

# define operations
a_ = np.asarray
r_ = np.r_


def SI_batch_resave(infile, outfile, nFr=-1, nFrChunk=300, rewriteOk=False, downscaleTuple=None):
    """
    :param infile: (str) path to infile
    :param outfile: (str) path to outfile
    :param nFr: (int) number of total frames to read from file, 0:nFr, if -1, then take nFr to be len(infile)
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
        if nFr is None or nFr == -1:
            T = len(tif.pages)
        else:
            T = nFr
            # note len(tif.pages) is slow on large files
            assert T <= len(tif.pages), 'number of frames in file is less than user-defined number to process'
        nR,nC = tif.pages[0].shape
    if downscaleTuple is not None:
        assert T % downscaleTuple[0] == 0, 'number of processed frames must be a multiple of downscaleTuple at z dimension'
        assert nFrChunk % downscaleTuple[0] == 0, 'number of frames per chunk must be multiple of downscaleTuple at z dimension'
        nR = int(nR/downscaleTuple[1])
        nC = int(nC/downscaleTuple[2])
    
    im = np.zeros((T//downscaleTuple[0],nR,nC), dtype='int16')
    print('Resaving. Frames done:')
    for fr in r_[0:T:nFrChunk]:
        if fr+nFrChunk <= T:
            ix = r_[fr:fr+nFrChunk]
        else:
            ix = r_[fr:T]
        chunk = tfl.imread(infile,key=ix)
        if downscaleTuple is not None:
            chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            if fr+nFrChunk <= T:
                fr_ds = fr//downscaleTuple[0]
                nFrChunk_ds = nFrChunk//downscaleTuple[0]
                ix = r_[fr_ds:fr_ds+nFrChunk_ds]
            else:
                fr_ds = fr//downscaleTuple[0]
                T_ds = T//downscaleTuple[0]
                ix = r_[fr_ds:T_ds]
        im[ix,:,:] = chunk.astype('int16')
        print(min(fr+nFrChunk,T), end=' ')
    
    print('\nSaving...', end='')
    tfl.imsave(outfile, im, bigtiff=True)
    print('Done. Saved to {}'.format(outfile))
    print('Output image size: ',im.shape)


def makePositive(im):
    if np.min(im) < 0:
        im += abs(np.min(im))
    return im


def dropStimFrs(rootdir,tifname,nFrsTrial,frsToDrop):
    im = tfl.imread(os.path.join(rootdir,tifname))
    nFr = im.shape[0]
    dropFrs = np.empty(0,dtype=int)
    for f in frsToDrop:
        dropFrs1 = dropFrs
        dropFrs2 = r_[f : nFr : nFrsTrial]
        dropFrs = np.sort(np.concatenate((dropFrs1,dropFrs2)))
    print(dropFrs)
    
    keepFrs = np.setdiff1d(r_[0:nFr], dropFrs)
    im_clean = im[keepFrs,:,:]
    print(im_clean.shape)

    tfl.imsave(os.path.join(rootdir,'imageFrames_stimCut.tif'),im_clean,bigtiff=True)


def rollingFrAvg(rootdir,tifname,avg_window):
    im = tfl.imread(os.path.join(rootdir,tifname))
    nFr = im.shape[0]
    
    assert nFr % avg_window == 0, 'the total number of frames must be a multiple of the window size for averaging'
    
    nFr_new = nFr // avg_window
    im_new = np.zeros((nFr_new,im.shape[1],im.shape[2]))
    for i in range(nFr_new):
        start_idx = i*avg_window
        end_idx = i*avg_window + avg_window
        avg_frame = np.mean(im[start_idx:end_idx,:,:],axis=(0))
        im_new[i] = avg_frame

    print(im_new.shape)

    tfl.imsave(os.path.join(rootdir,'imageFrames_rollAvg.tif'),im_new,bigtiff=True)


# optional preprocessing steps
def run_preprocessing(preprocessing_opts):
    '''run preprocessing steps as defined by user
    returns:
    file_name: name of preprocessed file to be used for caiman pipeline analysis
    mc_template_caiman: motion_correction_template for caiman pipeline
    '''
    
    # extract parameters
    file_path = preprocessing_opts['file_path']
    
    do_batch_resave = preprocessing_opts['do_batch_resave']
    SI_file_name = preprocessing_opts['SI_file_name']
    
    cut_stim_frames = preprocessing_opts['cut_stim_frames']
    frames_to_drop = preprocessing_opts['frames_to_drop']
    nFrTrial = preprocessing_opts['nFrTrial']
    
    use_coarse_mc = preprocessing_opts['use_coarse_mc']
    
    use_frame_avg = preprocessing_opts['use_frame_avg']
    avg_window = preprocessing_opts['avg_window']
    
    use_mc_template_caiman = preprocessing_opts['use_mc_template_caiman']
    template_path = preprocessing_opts['template_path']
    template_name = preprocessing_opts['template_name']
    
    # write params to text file
    out_prep_opts_name = os.path.join(file_path, "preprocessing_params.txt")
    param_textfile = open(out_prep_opts_name, 'wt')
    param_textfile.write(str(preprocessing_opts))
    param_textfile.close()
    
    
    # run preprocessing steps
    if do_batch_resave == 1:
        if not os.path.exists(os.path.join(file_path,'imageFrames_stim.tif')):
            print('Running batch_resave()...')
            infile_stim = os.path.join(file_path,SI_file_name)
            outfile_stim = os.path.join(file_path,'imageFrames_stim.tif')
            outfile_resave = os.path.join(file_path,'batch_resave.tif')
            downscale_factors = (1,2,2) #(t,x,y)

            SI_batch_resave(infile_stim,outfile_resave,downscaleTuple=downscale_factors,rewriteOk=True)
            im = makePositive(tfl.imread(outfile_resave))
            tfl.imsave(outfile_stim,im,bigtiff=True)
            os.remove(outfile_resave)
            print('Infile preprocessed and saved.')
            print('Batch resave done...\n')
    # let the default file_name be 'imageFrames_stim.tif' even if batch_resave 
    # is not performed (e.g. if testing different preprocessing steps)
    file_name = 'imageFrames_stim.tif'

    if cut_stim_frames == 1:
        if not os.path.exists(os.path.join(file_path,'imageFrames_stimCut.tif')):
            print('Dropping stim frames...')
            dropStimFrs(file_path,file_name,nFrTrial,frames_to_drop)
            print('Dropping stim frames done...\n')
        file_name = 'imageFrames_stimCut.tif'

    if use_coarse_mc == 1:
        if not os.path.exists(os.path.join(file_path,'imageFrames_coarse_mc.tif')):
            print('Running coarse motion correction...')
            im = tfl.imread(os.path.join(file_path,file_name))
            frames_to_align_to = r_[:1000]
            im_align = ptMH.image.align_stack(im,frames_to_align_to,do_plot=True)
            tfl.imsave(os.path.join(file_path,'imageFrames_coarse_mc.tif'),im_align,bigtiff=True)
            print('Coarse motion correction done...\n')
        file_name = 'imageFrames_coarse_mc.tif'

    if use_frame_avg == 1:
        if not os.path.exists(os.path.join(file_path,'imageFrames_rollAvg.tif')):
            print('Calculating rolling frame average...')
            rollingFrAvg(file_path,file_name,avg_window)
            print('Calculating rolling frame average done...\n')
        file_name = 'imageFrames_rollAvg.tif'

    if use_mc_template_caiman == 1:
        template_whole = tfl.imread(os.path.join(template_path,template_name))[:]
        mc_template_caiman = np.mean(template_whole,axis=0)
        print('Using template for motion correction...')
    else:
        mc_template_caiman = None
        print('Not using template for motion correction...')

    print('Preprocessing done. File to be analyzed: ',os.path.join(file_path,file_name))
    out_prep_opts_name = os.path.join(file_path, 'preprocessing_params.txt')
    param_textfile = open(out_prep_opts_name, 'wt')
    param_textfile.write(str(preprocessing_opts))
    param_textfile.close()
    
    return file_name,mc_template_caiman
