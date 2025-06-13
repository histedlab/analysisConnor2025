from . import _version

__version__ = _version.__version__
del _version


import packaging
from packaging.version import Version
import matplotlib as mpl

if Version(mpl.__version__) < Version("2.2.2"):
    raise RuntimeError("Bad matplotlib version: upgrade to 2.2.2 or later to avoid plotting bugs")
    # we found bugs with plt.contour and with changing markersizes in intrinsic maps
