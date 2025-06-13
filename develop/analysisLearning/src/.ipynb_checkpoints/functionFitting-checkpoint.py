import scipy.special as special
import scipy.optimize as opt
import numpy as np

# von mises pdf, plus scale parameter
def vmCustom(x,mu,kappa,scale,intercept):
    numerator = np.exp(kappa*np.cos(x-mu))
    denom = 2*np.pi*special.iv(0,kappa)
    
    return ((numerator/denom)*scale)+intercept

# constrained von mixes mixture, where mu2 = -mu1 and mix2 = 1-mix1
def vmCustomConstrainedMix(x,mu,kappa,scale,intercept,mix):
    vm1 = np.exp(kappa*np.cos(x-mu)) / (2*np.pi*special.iv(0,kappa))
    vm2 = np.exp(kappa*np.cos(x+mu)) / (2*np.pi*special.iv(0,kappa))
    
    return (scale*((vm1*mix)+(vm2*(1-mix))))+intercept

# fit von mises pdf, returns parameters and their variances
def fit_vmCustom(xdata,ydata):
    bounds = ([-np.pi,0,0,-np.inf],[np.pi,np.inf,np.inf,np.inf])
    paramFits = opt.curve_fit(vmCustom,xdata,ydata,bounds=bounds,max_nfev=1000)
    return paramFits[0],np.diag(paramFits[1])

# fit von mixes constrained mixture
def fit_vmCustomConstrainedMix(xdata,ydata):
    bounds = ([-np.pi,0,0,-np.inf,0],[np.pi,np.inf,np.inf,np.inf,1])
    paramFits = opt.curve_fit(vmCustomConstrainedMix,xdata,ydata,bounds=bounds,max_nfev=1000000)
    return paramFits[0],np.diag(paramFits[1])

# sample from any given custom pdf (or pdf like thing)
def generateSamples(pdfFunc,pdfParams,xMin,yMin,numSamples):
    uniSampling = np.random.uniform(xMin,yMin,size=(numSamples,))
    pdfSampling = pdfFunc(uniSampling,*pdfParams)
    
    return (np.random.choice(uniSampling,replace=True,p=pdfSampling/np.sum(pdfSampling),size=(numSamples,)))