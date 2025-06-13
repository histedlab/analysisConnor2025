from mworksbehavior import spikes
import numpy as np

def test_wf_fit_with_saved():
    wfV = np.load('./data/spike_wf_example.npz')['wfV']

    widthMs = spikes.get_width_from_mean(wfV, samprate=30000, debugPlot=False)
    # hardcoded width, if this changes check the output, with debugPlot true
    assert np.isclose(widthMs, 0.51, atol=1e-2, rtol=1e-2)
