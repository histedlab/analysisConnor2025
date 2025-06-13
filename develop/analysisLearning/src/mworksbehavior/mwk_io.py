"""This module contains all functions that depend on the MWorks mwk libraries"""

import sys, os
import pandas as pd
import warnings


try:
    mwPath = "/Library/Application Support/MWorks/Scripting/Python"
    if os.path.exists(mwPath):
        sys.path.insert(0, mwPath)
    import mworks
    from mworks.data import MWKFile
except ImportError:
    # this except allows this file to be imported, but code that requires mworks will not run w/out error
    warnings.warn("MWorks modules not found at %s. This code will not be able to read mwk2 files." % mwPath)


def mwk_to_h5(mwk_file, out_file=None, keep_system_vars=False, exist_delete=False):
    """This is the function formerly called by the script
    Args:
        exist_delete: boolean, or 'skip_create'.  'skip_create' means just return on exist
        out_file: if None, same as mwk_file with .h5 replacing .mwk/mwk2 extension"""

    if out_file is None:
        rootname, ext = os.path.splitext(mwk_file)
        out_file = rootname + ".h5"

    if os.path.exists(out_file):
        if type(exist_delete) is str and exist_delete == "skip_create":
            return
        elif exist_delete == True:
            os.unlink(out_file)
        elif exist_delete == False:
            raise RuntimeError("Output file %s exists: delete and try again" % out_file)
        else:
            raise RuntimeError(f"bad value for exist_delete: {exist_delete}")

    outL = []
    with MWKFile(os.fspath(mwk_file)) as f:
        codec = dict(f.codec)
        events = f.get_events()
        allCodesWNames = f.reverse_codec.values()

        for i, evt in enumerate(events):
            if evt.code in allCodesWNames:
                tagname = codec[evt.code]
                if not keep_system_vars and tagname[0] == "#":
                    continue
                outL.append((tagname, evt.time, evt.value))

    df = pd.DataFrame(outL, columns=("tagname", "timeUs", "value"))
    df.timeUs = df.timeUs.astype("int64")
    # MH191017: not sure if the code stream is out of order in mwkfile, or shuffled above, but
    # it's ok to sort here and represent always as in order of timeUs
    df = df.sort_values("timeUs").reset_index(drop=True)

    # write output
    with warnings.catch_warnings():
        # filter the 'performancewarning'.  See https://github.com/pandas-dev/pandas/issues/3622.
        # we could try setting the store option to 'table', but I don't think it's very important
        warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
        df.to_hdf(out_file, key="mwk_events", mode="w")
