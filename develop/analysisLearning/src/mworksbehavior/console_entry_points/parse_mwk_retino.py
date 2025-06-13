#!/usr/bin/env python3


########
# TODO: add in user argument for selecting stim range for potential buggy mwk2 files


########
import os,sys
import shutil
import numpy as np
from mworksbehavior import mwkfiles,mwk_io
import mworksbehavior.imaging as mwb_im
import pickle
import argparse
from pathlib import Path
import logging
print(__name__)
logger = logging.getLogger(__name__)


### setup cmdline args
def parse_args(argv):
    def_input_dir = Path('~/Documents/MWorks/Data').expanduser()
    def_output_dir = Path('/Volumes/HistedLab/TwoPhotonImages')

    parser = argparse.ArgumentParser()
    parser.add_argument('input_name', type=str,
        help=f'Input mwk2 file. If path not specified, use {def_input_dir}')
    parser.add_argument('output_path', nargs='?', type=str, default=def_output_dir,
        help=f'Output path. Output is pkl of data dict. If not specified, put on server at {def_output_dir}')
    parser.add_argument('--paulname', action='store_true',
        help='Autocompute a child output file directory from input filename: must have format yymmdd-innnn-stim*')
    args = parser.parse_args(argv)

    # some processing for input file
    print(args)
    inp = Path(args.input_name)
    if inp.parent == Path('.'): # no dir specified
        inp = def_input_dir / inp
    if not inp.is_file():
        raise OSError(f'.mwk2 filepath not found: {inp}')
    args.inputP = inp
    logger.info(f'Input file: {args.inputP}')

    # process some things about output file
    args.outputDirP = Path(args.output_path)
    if not args.outputDirP.is_dir():
        raise OSError(f'output path {args.outputDirP} not found. Is network drive mounted?')

    if args.paulname:
        # create custom directory name for saving output on network drive (works for PKL's file-naming convention, see help)
        filename_parts = list(args.inputP.stem.split('-'))
        # check for naming convention
        assert len(filename_parts) == 3
        assert filename_parts[1][0] == 'i'
        assert len(filename_parts[0]) == 6
        assert filename_parts[2][:4] == 'stim'
        custom_dirname = filename_parts[0]+'-'+filename_parts[1]
        args.outputDirP = args.outputDirP / custom_dirname
        args.outputDirP.mkdir(exist_ok=True) # make this new directory - note above checks the parent dir is already parent
        logger.info(f'Created directory {args.outputDirP}')

    return args


######## utility functions for handling mwk2 files and mwf object
def load_mwk_output(mwk_filepath,useStimRange='all'):
    '''load the .mwk file to extract mworks experiment parameters'''
    # read mworks file
    try:
        mwf = mwkfiles.RetinotopyMap2StimMWKFile(mwk_filepath,stims_to_keep=useStimRange)
    except:
        mwf = mwkfiles.RetinotopyMap1MWKFile(mwk_filepath,stims_to_keep=useStimRange)
    mwf.compute_imaging_constants()
    print(mwf.constS)
    print('nStims: %d, nFramesPerStim: %d, nReps: %d' % (mwf.nstim, mwf.nframes_stim, mwf.nreps))
    return mwf

def parse_mwk_levels(mwf):
    '''generate a list containing lists of frames for each stim level for an MWorks experiment'''
    levels = np.unique(mwf.stimDf[mwf.levelVar])
    levelFr = np.zeros((len(levels),int(mwf.nframes_stim*mwf.nreps)))
    for (iL,tL) in enumerate(levels):
        desIdx = np.where(mwf.stimDf[mwf.levelVar].values==tL)[0]
        levelFr_idx = []
        for idx in desIdx:
            levelFr_idx+=(list(range(mwf.nframes_stim*idx,mwf.nframes_stim*idx+(mwf.nframes_stim))))
        levelFr[iL,:] = np.asarray(levelFr_idx)
    levelFr = levelFr.astype(int)
    nLevel = len(levels)
    return levels,levelFr,nLevel


######## other util functions
def yes_no(prompt):
    reply = None
    while reply not in ("y", "n"):
        reply = input(prompt).lower()
    return (reply == "y")



def entry():
    """Function that gets called by the script: see setup.py entry point. Calls main() w/ sys.argv
    We do it this way to let main be called with arbitrary argv for testing"""
    main(sys.argv[1:])

def main(argv):
    """Function that does the script work.
    Args:
        argv: vector of commandline arguments, equiv to sys.argv[1:]
    """

    # we are being called as a script, so setup logging here
    logging.basicConfig(format='-- %(message)s', level=logging.DEBUG)


    args = parse_args(argv)

    # copy .mwk2 to output location
    shutil.copy2(args.inputP,args.outputDirP)
    logger.info(f'{args.inputP.name} copied to {args.outputDirP}')

    mwk2_filepath = args.outputDirP / args.inputP.name

    # read in mwk2 file
    print('Loading .mwk2 file...')
    mwf = load_mwk_output(mwk2_filepath.as_posix())

    # run mwf object through parsing function to extract which frames correspond
    # to which trial type (level)
    levels,levelFr,nLevel = parse_mwk_levels(mwf)

    # grab output from mwf and parsing function
    nPreFr = mwf.constS.counterNPre
    nStimFr = mwf.constS.counterNStim
    nPostFr = mwf.constS.counterNPost
    nTrialLevel = int(mwf.nreps)
    nFrTrial = int(mwf.nframes_stim)

    # create dictionary of all variables to save as output
    out_dict = {
        'nPreFr': nPreFr,
        'nStimFr': nStimFr,
        'nPostFr': nPostFr,
        'nTrialLevel': nTrialLevel,
        'nFrTrial': nFrTrial,
        'nLevel': nLevel,
        'levels': levels,
        'levelFr': levelFr
    }

    # save dictionary into a pickle file at specified location
    save_pkl_filepath = mwk2_filepath.parent / (mwk2_filepath.stem + '-mwk2.pkl')
    pickle.dump(out_dict,open(save_pkl_filepath, 'wb'))
    logger.info(f'.mwk2 extracted parameters saved in pickled dict at: \n {save_pkl_filepath}')
    logger.info('Done.')
