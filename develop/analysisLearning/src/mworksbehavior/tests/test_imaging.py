import numpy as np
import tifffile as tfl
from mworksbehavior.imaging import io as imio
from pathlib import Path
r_ = np.r_





def test_downscale_with_batch_resave(tmp_path):
    fname = Path('data/test_5fr_tiff_fijiout_256x256.tif').expanduser()
    outname = tmp_path / 'outfile.tif'

    imio.SI_batch_resave(fname, outname, nFr=-1, nFrChunk=5, rewriteOk=True, downscaleTuple=(1, 2, 2))

    outdata = tfl.imread(outname)
    assert np.all(outdata.shape == r_[5, 128, 128])

def test_downscale_z_dimension_with_batch_resave(tmp_path):
    fname = Path('data/test_5fr_tiff_fijiout_256x256.tif').expanduser()
    outname = tmp_path / 'outfile.tif'

    imio.SI_batch_resave(fname, outname, nFr=4, nFrChunk=4, rewriteOk=True, downscaleTuple=(2, 1, 1))

    outdata = tfl.imread(outname)
    assert np.all(outdata.shape == r_[2, 256, 256])