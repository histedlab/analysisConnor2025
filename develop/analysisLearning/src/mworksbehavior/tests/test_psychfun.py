import unittest
from unittest import TestCase

import warnings
warnings.filterwarnings("ignore", "Using or importing the ABCs from 'collections'") # DeprecationWarning
warnings.filterwarnings("ignore", "zmq.eventloop.ioloop is deprecated in pyzmq 17") # DeprecationWarning

import numpy as np
import pandas as pd
import seaborn as sns
import os,sys
import subprocess as sp
import scipy.io
import pytoolsMH as ptMH
import collections
import shutil
import pytest

from mworksbehavior import mat_io, psychfun


class Prelims(unittest.TestCase):
    def check_python_ver(self):
        self.assertTrue(sys.version_info[0] >= 3, msg='This whole suite depends on python 3 (in part because of newer pandas features))')

    def check_pandas(self):
        d = pd.DataFrame()
        import pdb; pdb.set_trace() 
        self.assertTrue(hasattr(d, 'transform'), msg='Needs a later pandas version')


class TestPsychfun(unittest.TestCase):
    discardResultXlsName = './tmp/to-delete_results.xlsx' # have to write xlsx, xlwt deprecated 240123
    discardResultXlsNameRange = './tmp/to-delete_results_range.xlsx'
    discardPdfDir = './tmp/pdfOutputs'
    
    @pytest.mark.filterwarnings("ignore:Using or importing the ABCs")
    def test_do_many_fits(self):
        # First, init object
        paramXlsName = './data/psychfun_fit_params.xls'
        paramXlsNameRange = './data/psychfun_fit_params2-simple-range.xls'
        localDataDir = '../../test_data'

        if os.path.exists(self.discardResultXlsName):
            os.remove(self.discardResultXlsName)
        if os.path.exists(self.discardResultXlsNameRange):
            os.remove(self.discardResultXlsNameRange)
        os.makedirs('./tmp', exist_ok=True)
        mffp = psychfun.ManyFitsFromParams(paramXlsName,
                                           self.discardResultXlsName, localDataDir)
        mffpr = psychfun.ManyFitsFromParams(paramXlsNameRange,
                                           self.discardResultXlsNameRange, localDataDir)
        
        # double check power, a few other params, are as read previously
        mffp.do_all_fits(nBootstrapReps=3, redoAll=True)
        mffpr.do_all_fits(nBootstrapReps=3, redoAll=True)

        mffp.do_all_plots(self.discardPdfDir,redoAll=True)

        columns = ['Threshold1', 'ThreshY1']
        self.assertFalse(mffp.resultDf.loc[(1230, '171016', ''),columns].equals(mffpr.resultDf.loc[(1230, '171016', ''),columns]),
                         msg='Range Test Failed fit to subrange gives same fit values as all data')
        
    def tearDown(self):
        if os.path.exists(self.discardResultXlsName):
            os.remove(self.discardResultXlsName)
        if os.path.exists(self.discardPdfDir):
            shutil.rmtree(self.discardPdfDir)

        
if __name__ == '__main__':
    unittest.main()
