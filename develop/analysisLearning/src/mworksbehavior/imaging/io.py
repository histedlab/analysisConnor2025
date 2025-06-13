import os
import logging
import tifffile # requires mh37e, and may work with earlier
import tifffile as tfl
import numpy as np
from skimage import transform
r_ = np.r_


log = logging.getLogger(__name__)


def SI_batch_resave(infile, outfile, nFr=-1, nFrChunk=300, rewriteOk=False, downscaleTuple=None):
    """Resave ScanImage tiffs into FIJI-readable tiffs, with optional resize.
    Uses as much memory as needed by the *output* file.

    Args:
        infile: (str) path to infile
        outfile: (str) path to outfile
        nFr: (int) number of total frames to read from file, 0:nFr, if -1/None, then take nFr to be len(infile)
        nFrChunk: (int) number of frames to load into memory at once
        rewriteOk: (bool) whether or not to overwrite existing file
        downscaleTuple: (tuple) downscaling factor in (z, x, y), e.g. (1, 2, 2) for 2x downscale
    Returns:
        nothing, just writes output file

    Notes:
    - MH 201020: consumes the amount of memory needed by the _output_ file.  Input file read one frame at a time.
    - created by Anna Li, edited by PKL, MH

    TODO:
    - should update to use logging rather than print statements... someday. ask MH
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
        nR, nC = tif.pages[0].shape
    if downscaleTuple is not None:
        assert T % downscaleTuple[0] == 0, 'num processed frames must be a multiple of downscaleTuple at z dim'
        assert nFrChunk % downscaleTuple[0] == 0, 'num frames per chunk must be multiple of downscaleTuple at z dim'
        nR = int(nR / downscaleTuple[1])
        nC = int(nC / downscaleTuple[2])

    im = np.zeros((T // downscaleTuple[0], nR, nC), dtype='int16')
    for fr in r_[0:T:nFrChunk]:
        if fr + nFrChunk <= T:
            ix = r_[fr:fr + nFrChunk]
        else:
            ix = r_[fr:T]
        chunk = tfl.imread(infile, key=ix)
        if downscaleTuple is not None:
            chunk = transform.downscale_local_mean(chunk, downscaleTuple)
            if fr + nFrChunk <= T:
                fr_ds = fr // downscaleTuple[0]
                nFrChunk_ds = nFrChunk // downscaleTuple[0]
                ix = r_[fr_ds:fr_ds + nFrChunk_ds]
            else:
                fr_ds = fr // downscaleTuple[0]
                T_ds = T // downscaleTuple[0]
                ix = r_[fr_ds:T_ds]
        im[ix, :, :] = chunk.astype('int16')
        print(min(fr + nFrChunk, T), end=' ')

    print('\nSaving...', end='')
    tfl.imwrite(outfile, im, bigtiff=True)
    print('Done. Saved to {}'.format(outfile))
    print('Output image size: ', im.shape)


def make_positive(im):
    if np.min(im) < 0:
        im += abs(np.min(im))
    return im


def _example_load_tiff(file_name, subindices=None):
    """Wrapper around tifffile to load the different kinds of tif files we generate
    Code taken mostly from caiman - 190525

    201020: this is just example code now to read multipage and non-multipage tiffs
    (my guess is Caiman has code to handle them differently for a good reason they found?)

    TODO: probably best to create a load_tiff function, tested on scanimage and fiji and Pythong tiffs
        Or just use caiman's?  Or improve tifffile directly?
    Args:
        file_name:
        subindices: pages from tiff
    Returns:
        stack
    """
    # inspired by caiman.load()
    _, extension = os.path.splitext(file_name)[:2]
    extension = extension.lower()
    if extension == '.tif' or extension == '.tiff':
        with tifffile.TiffFile(file_name) as tffl:
            multi_page = True if tffl.series[0].shape[0] > 1 else False
            if len(tffl.pages) == 1:
                log.warning('Your tif file is saved a single page' +
                                'file. Performance will be affected')
                multi_page = False
            if subindices is not None:
                if type(subindices) is list:
                    if multi_page:
                        input_arr = tffl.asarray(key=subindices[0])[:, subindices[1], subindices[2]]
                    else:
                        input_arr = tffl.asarray()
                        input_arr = input_arr[subindices[0], subindices[1], subindices[2]]
                else:
                    if multi_page:
                        input_arr = tffl.asarray(key=subindices)
                    else:
                        input_arr = tffl.asarray()
                        input_arr = input_arr[subindices]

            else:
                input_arr = tffl.asarray()

            input_arr = np.squeeze(input_arr)
    return input_arr
