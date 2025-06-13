"""This module tests h5 -> MWKFile parsing
"""
import numpy as np
import warnings
import pytest
import os
import sys
from pathlib import Path

r_ = np.r_

# none of below works 200719
# pytestmark = pytest.mark.filterwarnings("error")
# warnings.filterwarnings("ignore")

from mworksbehavior import mwkfiles

### Info
#
# mwk2 vs h5, 241226:
#   - some of the early tests here save h5 files because they are smaller than corresponding mwk files.
#     I believe mwk2 files use compression, so they are often smaller than h5 files. In 2024 new tests save mwk2 files into
#     data/, not h5 files (work from PKL)
#

serverroot = "data"

def fulln_pandas_bug_convert_200719(fulln):
    """Workaround for pandas bug  https://github.com/pandas-dev/pandas/issues/33186
    Remove this function after the bug is fixed"""
    pn = Path(fulln).resolve()
    if pn.suffix == ".h5":
        return (pn.parent / "mwk_pickles" / pn.stem).with_suffix(".pkl.gz")
    else:
        return fulln


# suppress useless warnings see https://github.com/numpy/numpy/issues/11788
pytestmark = pytest.mark.filterwarnings("ignore: numpy.ufunc size changed")


def test_python_ver():
    assert sys.version_info[0] >= 3, "This whole suite depends on python 3 (in part because of newer pandas features))"


## RM2, 2019, laser and vis


def test_sept2019_retinomap2stim():
    fulln = Path(serverroot) / "i1974-190910-Stim0.h5"
    mwf = mwkfiles.RetinotopyMap2StimMWKFile(str(fulln))
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    vs = ([0], [1], [0, 0.5, 1, 5])
    for iV, vn in enumerate(["tStim1ElevationDeg", "tStim1Contrast", "tAPeakPowerMw"]):
        tV = mwf.stimDf.loc[:, vn]
    assert np.all(np.equal(vs[iV], np.unique(tV)))

    assert len(mwf.stimDf) == 120
    assert mwf.constS.counterNPre == 10

    # Retinomap2stim has stim announce _before_ first pre frame, make sure parsed in the correct order
    assert np.all(mwf.stimDf.loc[0:3, "stimPySelLevel"].to_numpy() == [3, 2, 0, 1])
    assert np.all(mwf.stimDf.loc[116:119, "stimPySelLevel"].to_numpy() == [3, 2, 0, 1])  # coincidentally matches above

    # now parse it a second time to check to make sure exptId checking is working
    def unfixFn(df0):
        eIx = df0.tagname == "experimentXmlTrialId"
        df0.loc[eIx, "value"] = 999  # break it
        return df0

    with pytest.raises(mwkfiles.CorruptFileError) as exc_info:
        mwf_shouldfail = mwkfiles.RetinotopyMap2StimMWKFile(str(fulln), fixDfFn=unfixFn)
    assert "Looking for exptXmlTrialIds 23, but found 999" in exc_info.value.args[0]


## grating files


def test_oct2019_file_with_extra_three_starts():
    """Apparently due to"""
    fulln = os.path.join(serverroot, "190922_i2035_stim2.h5")
    mwf = mwkfiles.RetinotopyMap1MWKFile(fulln, use_counter_as_start=True)  # , doTryFixCorrupt=False)
    mwf.compute_imaging_constants()

    assert len(mwf.stimDf) == 500
    df0 = mwf.get_codedf("counterFIO")
    assert mwf.get_const_values("counterNPre").to_numpy() == 7
    assert np.all(mwf.get_const_values(("counterNPost", "counterNStim")).to_numpy() == (7, 1))
    # 7507 counter ticks: 3 from last file (discarded by startUs), 4 extra at end (after last end run),
    # and then all 0, 1...7500 ticks inside file - 250120 now fixes this by using correct start and end times
    assert len(df0) == 7500, "bug?  no longer matches? Was 7501 before 250120 counter fixes"


def test_sep2019_file_with_two_not_three_first_ends():
    fulln = os.path.join(serverroot, "i1974_190905_stim1.h5")
    mwf = mwkfiles.ChRMap1MWKFile(fulln, doTryFixCorrupt=True)

    mwf.compute_imaging_constants()

    assert len(mwf.stimDf) == 50
    # note 250120 changes: the zero counter tick is dropped, so 4002 if we wanted to check this
    #df0 = mwf.get_codedf("counterFIO", only_in_startend_times=False)
    #assert len(df0) == 4003, "there are 4000 counter ticks in the file, first three are discarded"
    df0 = mwf.get_codedf("counterFIO", only_in_startend_times=True)
    assert len(df0) == 4000, "there are 4000 counter ticks in the file, first three are discarded"


def test_nov17_file_with_missing_start():
    fulln = os.path.join(serverroot, "171117-i1230-stim0.h5")
    fulln = fulln_pandas_bug_convert_200719(fulln)  # REMOVE AFTER BUGFIX
    mwf = mwkfiles.RetinotopyMap1MWKFile(fulln, firstTrStartTimeUs=0, use_counter_as_start=True)  # defaults
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    assert len(df0) == 962 - 1 # 250120 changes, drops the zero tick
    assert len(mwf.stimDf) == 60


def test_nov17_file2():
    fulln = os.path.join(serverroot, "171117-i1230-stim3.h5")
    fulln = fulln_pandas_bug_convert_200719(fulln)  # REMOVE AFTER BUGFIX
    mwf = mwkfiles.RetinotopyMap1MWKFile(
        fulln, firstTrStartTimeUs=None, use_counter_as_start=False
    )  # auto compute firstst
    mwf.compute_imaging_constants()

    #df0 = mwf.get_codedf("counterFIO", only_in_startend_times=False)
    #assert len(df0) == 725
    df0 = mwf.get_codedf("counterFIO", only_in_startend_times=True)
    assert len(df0) == 720
    assert len(mwf.stimDf) == 45


@pytest.mark.filterwarnings("ignore:Found two counter values")
def test_aug2018_withtwostarts():
    fulln = os.path.join(serverroot, "i1716_180808_hemo_stim0.h5")
    fulln = fulln_pandas_bug_convert_200719(fulln)  # REMOVE AFTER BUGFIX
    mwf = mwkfiles.RetinotopyMap1MWKFile(fulln, firstTrStartTimeUs=None, use_counter_as_start=True)
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    df0 = mwf.get_codedf("counterFIO", only_in_startend_times=False)
    assert len(df0) == 7265 - 1 # 250120 changes, drops the zero tick
    df0 = mwf.get_codedf("counterFIO", only_in_startend_times=True)
    assert len(df0) == 7262

    assert len(mwf.stimDf) == 180


## ChrMap files


def test_dec11_file_chrmap():
    """First file we used"""
    fulln = os.path.join(serverroot, "i1360_171211_stim0.h5")
    fulln = fulln_pandas_bug_convert_200719(fulln)  # REMOVE AFTER BUGFIX
    mwf = mwkfiles.ChRMap1MWKFile(fulln)
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    assert len(mwf.stimDf) == 100
    assert np.all(mwf.stimDf.tLaserPySelLevel[-4:] == r_[3, 0, 2, 1])
    assert mwf.constS.counterNPreTrig == 15


def test_dec14_file_chrmap():
    """After fixing extra-stim bug (tPySelLevel set before counterFIO==1)"""
    fulln = os.path.join(serverroot, "i1360_171214_stim0.h5")
    fulln = fulln_pandas_bug_convert_200719(fulln)  # REMOVE AFTER BUGFIX
    mwf = mwkfiles.ChRMap1MWKFile(fulln)
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    assert len(mwf.stimDf) == 60
    assert np.all(mwf.stimDf.tLaserPySelLevel[-4:] == r_[0, 2, 1, 0])
    assert mwf.constS.counterNPreTrig == 30


def test_jan2020_file_chrmap():
    """First three start codes longer than 5 ms, throws an error that can be fixed with doTryFixCorrupt"""
    fulln = os.path.join(serverroot, "200120-i2415-stim1.h5")
    mwf = mwkfiles.ChRMap1MWKFile(fulln, doTryFixCorrupt=True)
    mwf.compute_imaging_constants()
    df0 = mwf.get_codedf("counterFIO")

    assert len(mwf.stimDf) == 4
    assert np.all(mwf.stimDf.tAPeakPowerMw == r_[0.2, 2.0, 2.0, 0.2])
    assert mwf.constS.counterNPreTrig == 90


def test_aug2021_loadonemwk2():
    fulln = (Path(serverroot) / "210725-i3689-stim2.mwk2").resolve()
    # remove the h5 file if it exists
    h5P = fulln.with_suffix(".h5")
    if h5P.is_file():
        h5P.unlink()
    # let the mwkFile object create the h5 file - must have MWorks installed, only on a mac
    mwf = mwkfiles.RetinotopyMap1MWKFile(fulln)
    mwf.compute_imaging_constants()
    assert mwf.constS.counterNPre == 60


### tests here and below use mwk2 files not h5 in data/, because mwk2 files are smaller


def test_dec2024_mwk2counter_issue1():
    '''after first 3 counter codes, the next counter code for 0 does not show up;
    issue1: the first 3 counter codes are 0 (i.e. mworks newly opened, start button unpressed prior to collection).
    The code now handles this without error: it detects the first 1 value and end of last trial.'''
    fulln = os.path.join(serverroot, "240918-countertest-issue1.mwk2")
    mwf = mwkfiles.RetinotopyMap2StimMWKFile(fulln)
    mwf.compute_imaging_constants()
    assert mwf.counterStats.nCounterTicks == 1920

@pytest.mark.filterwarnings("ignore:Trying to fix.*first trial is short, be careful")
@pytest.mark.filterwarnings("ignore:Found first counter tick to be 1, expected zero; subtracting 1")
def test_dec2024_mwk2counter_issue2():
    '''after first 3 counter codes, the next counter code for 0 does not show up;
    issue2: the first 3 counter codes are NOT 0 (i.e. start button was pressed prior to this collection, e.g. if
    a previous run had been collected and mworks not restarted)'''
    fulln = os.path.join(serverroot, "240918-countertest-issue2.mwk2")
    mwf = mwkfiles.RetinotopyMap2StimMWKFile(fulln, doTryFixCorrupt=True)
    mwf.compute_imaging_constants()
    assert mwf.counterStats.nCounterTicks == 1919 # not 1920, one missing code, see LabArchives/Mark and github issue #52
    assert mwf.counterStats.nTotalStims == 8

@pytest.mark.filterwarnings("ignore:Trying to fix.*first trial is short, be careful")
@pytest.mark.filterwarnings("ignore:Found first counter tick to be 1, expected zero; subtracting 1")
def test_dec2024_mwk2counter_issue3():
    '''after first 3 counter codes, the next counter code for 0 does not show up;
    issue2: the first 3 counter codes are 0 (i.e. mworks newly opened, start button unpressed prior to collection)
    AND the time between the first counter with code 1 and the previous counter with code 0 (the last counter code
    of the initial 3 at start) is too short, and the levelVar is not set in time (typically happens prior to counter
    code being set to 1); see line 371 in mwkfiles.py for location of error'''
    fulln = os.path.join(serverroot, "241111-countertest-issue3.mwk2")
    mwf = mwkfiles.RetinotopyMap2StimMWKFile(fulln, doTryFixCorrupt=True)
    mwf.compute_imaging_constants()
    assert mwf.counterStats.nCounterTicks == 38399 # not 38400, one missing
    assert mwf.counterStats.nTotalStims == 160

