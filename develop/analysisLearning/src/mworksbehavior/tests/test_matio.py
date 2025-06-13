import unittest
from unittest import TestCase
import numpy as np
import pytest  # this is the future

from mworksbehavior import mat_io

class TestFileRead(unittest.TestCase):

    def setUp(self):
        # read a test file
        fName = '../../test_data/data2-i1114-170406.mat'
        self.mbF = mat_io.matBehavFile(fName)

    def test_attribs(self):
        self.assertTrue(len(self.mbF.df) == 380)

        # double check power, a few other params, are as read previously
        self.assertEqual(np.sum(self.mbF.df.loc[:,'tLaserPowerMw']),
                        pytest.approx(21.497057728187805))
        self.assertEqual(np.sum(self.mbF.df.loc[:,'trialOutcomeCell']=='success'),
                         247)

        # any other stuff to check should be in here
        #import pdb;pdb.set_trace()

class TestMultiblockRead(unittest.TestCase):

    def setUp(self):
        # read a test file
        fName = '../../test_data/data2-i1081-171012.mat'
        self.mbF = mat_io.matBehavFile(fName)

    def test_attribs(self):
        self.assertTrue(len(self.mbF.df) == 400)


        # any other stuff to check should be in here
        #import pdb;pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
