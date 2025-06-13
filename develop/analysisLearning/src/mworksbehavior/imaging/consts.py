"""180509: unclear whether we need these constants"""

import os
import sys
import glob
import re
import numpy as np
from argparse import Namespace
import tifffile
import joblib
import warnings
r_ = np.r_
a_ = np.asarray



# todo
# - may wish to factor out common elements of DataDir2p and DataDir1p into a base class

# class for converting cnm_dict in an object
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class DataDir2p():
    """Contains constants about directories, files, figs, analysis outputs, etc.
    Also has simple reader functions

    Args
        subdir: default 'analysisCaiman': use for debugging only
    """
    def __init__(self, rootdir, mwkname=None, subdir=None, stackout_subdir=None):

        subdir = 'analysisCaiman' if subdir is None else subdir  # set defaults
        if not os.path.isdir(rootdir):
            raise RuntimeError('rootdir not found: %s' % rootdir)
        self.rootdir = rootdir

        if mwkname is None:
            a0 = glob.glob(os.path.join(rootdir, '*.mwk*'))
            assert (len(a0) == 1), 'Not just one .mwk in rootdir %s, found: %s' % (rootdir, a0)
            self.mwkname = a0[0]
        else:
            self.mwkname = mwkname

        if not os.path.exists(self.mwkname):  # note mwk can be changed to a directory (caching) by MWorks reader code
            raise RuntimeError('mwk not found: %s' % self.mwkname)

        self.mwkbasename, _ = os.path.splitext(os.path.basename(self.mwkname))

        # mwk h5 stuff
        self.h5name = re.sub(r'\.mwk[2]?$', '.h5', self.mwkname) # use regex to support mwk or mwk2 files
        self.h5stimsname = re.sub(r'\.mwk[2]?$', '-stims.h5', self.mwkname)

        # caiman analysis
        self.caiman_dir = os.path.join(self.rootdir, subdir)
        self.caiman_results_v1 = os.path.join(self.caiman_dir, 'results-analysis.npz') # old format, can be later removed
        self.caiman_results_v2 = os.path.join(self.caiman_dir, 'results-analysis-v2.gz') # new v2 format, 180618
        self.caiman_results_v3 = os.path.join(self.caiman_dir, 'results-analysis-v3.npz') # latest format for save 190610
        self.mc_stack_name = os.path.join(self.caiman_dir, 'imageFrames-mc.tif')

        # stack analysis
        if stackout_subdir is None:
            stackout_subdir = 'analysisStacks'
        self.stackout_dir = os.path.join(self.rootdir, stackout_subdir)
        os.makedirs(self.stackout_dir, exist_ok=True)
        self.mapfigname = os.path.join(self.stackout_dir, self.mwkbasename+'-map.pdf')


    def get_caiman_results(self):
        '''Loads in caiman results data file based on most recent save version present'''
        if os.path.exists(self.caiman_results_v3):
            cnm_dict = np.load(self.caiman_results_v3,allow_pickle=True)['results_dict'].item()
            return objectview(cnm_dict)
        elif os.path.exists(self.caiman_results_v2):
            # use a fake caiman package so we don't need to have it installed
            tP,_ = os.path.split(__file__)
            desP = os.path.join(tP, 'fake_packages')
            if not desP in sys.path:
                sys.path.append(desP)
            cnm_data = joblib.load(self.caiman_results_v2)
            # backward compat, load cnm fields into the base object
            for tn in ['A','C', 'S', 'b', 'f', 'YrA']:
                setattr(cnm_data, tn, getattr(cnm_data.cnm, tn))
            cnm_data.Cn = cnm_data.localCorr
            return cnm_data
        else:
            warnings.warn('New-format caiman save not found, using old V1 format, may be missing some data/params')
            cnm_data = Namespace(**dict(np.load(self.caiman_results_v1)))
            cnm_data.localCorr = cnm_data.Cn # back compat
            return cnm_data


    def get_output_stack(self, tifname):
        print(self.stackout_dir)
        return tifffile.imread(os.path.join(self.stackout_dir, tifname))
