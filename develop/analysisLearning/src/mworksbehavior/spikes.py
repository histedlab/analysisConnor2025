from skmisc.loess import loess  # 200110 status: pip install scikit-misc

import numpy as np
import pandas as pd

r_ = np.r_


def get_width_from_mean(meanWf, samprate=30000, debugPlot=False):

    xsMs = r_[0 : len(meanWf)] / samprate * 1000
    xsHat = np.linspace(0, 1.5, 200)

    l = loess(xsMs, meanWf, span=0.1, degree=1)  # degree=1 is lowess
    l.fit()
    pred = l.predict(xsHat, stderror=True)
    yHat = pred.values

    # find the biggest valley, then peak after the valley
    valleyN = np.argmin(yHat)
    valleyMs = xsHat[valleyN]
    newV = yHat.copy()
    newV[0:valleyN] = 0.0
    maxN = np.argmax(newV)
    peakMs = xsHat[maxN]

    if debugPlot:
        plt.plot(xsMs, meanWf, ".-")
        plt.plot(xsHat, yHat)
        conf = pred.confidence()
        plt.fill_between(xsHat, conf.lower, conf.upper, alpha=0.33)
        plt.plot(r_[valleyMs, peakMs], r_[1, 1] * np.min(wfV) * 1.1, "k")
        plt.xlabel("time (ms)")

    return peakMs - valleyMs


def get_all_widths_neo(blackrockFile, blackrockSegment):
    """
    Args:
        blackrockFile:
        blackrockSegment:

    Notes:
        Call neo code as:
            blackrockFile = neo.io.BlackrockIO(nevname.as_posix())
            blackrockSegment = br.read_segment()

    Returns:
        df: data frame with wire, unit, widthMs fields
    """
    rowL = []
    for (iSt, tSt) in enumerate(blackrockSegment.spiketrains):
        a0 = tSt.annotations
        (wire, unit) = a0["channel_id"], a0["unit_id"]
        wfs = blackrockFile.get_spike_raw_waveforms(unit_index=iSt)
        wfV = wfs.mean(axis=(0, 1))

        widthMs = get_width_from_mean(wfV, debugPlot=False)
        rowL.append({"wire": wire, "unit": unit, "widthMs": widthMs})
        # print(f'width: {widthMs:.3g}ms', end=', ')
    df = pd.DataFrame(rowL)
    return df
