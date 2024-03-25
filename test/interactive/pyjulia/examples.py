####
#### Start Julia and load DECAES
####
#### NOTE:
####
####    This script requires the julia package to be installed.
####    See the following documentation for instructions:
####
####        https://github.com/JuliaPy/pyjulia#quick-usage
####

import decaes
decaes.initialize()
from decaes import DECAES

####
#### Example using mock image for proof-of-concept demonstration
####

import numpy as np

# Create mock image
TE     = 10e-3 # arbitrary mock echo time
nTE    = 32 # arbitrary mock number of echoes
T2     = 80e-3 # true T2 value
SNR    = 60.0 # simulated signal-to-noise ratio
image  = np.exp(-np.linspace(TE, nTE*TE, nTE) / T2) # exponentially decaying signal
image  = np.tile(image.reshape((1, 1, 1, nTE)), (3, 3, 3, 1)) # reshape and repeat into shape (3, 3, 3, nTE) image
z_real = 10.0**(-SNR/20) * np.random.randn(*image.shape) # small amount of random noise (real channel)
z_imag = 10.0**(-SNR/20) * np.random.randn(*image.shape) # small amount of random noise (imaginary channel)
image  = np.sqrt((image + z_real)**2 + z_imag**2) # add Rician noise

# Call T2mapSEcorr to compute T2 distribution
t2maps, t2dist = DECAES.T2mapSEcorr(image, TE = TE, nT2 = 40, T2Range = (TE, 1.0), Reg = "lcurve")

# Call T2partSEcorr to compute T2 parts such as MWF (aka short fraction "sfr"), LWF (aka long fraction "mfr")
t2parts = DECAES.T2partSEcorr(t2dist, T2Range = (TE, 1.0), SPWin = (TE, 25e-3), MPWin = (25e-3, 200e-3))

msg = """
Mock image inference results:
GGM T2        = {ggm} (true = {T2})
Flip Angle    = {alpha} (true = 180.0)
SNR           = {snr} (true = {SNR})
SFR (aka MWF) = {sfr} (true = 0.0)
MFR (aka LWF) = {mfr} (true = 1.0)
""".format(
    T2    = T2,
    SNR   = SNR,
    ggm   = t2maps["ggm"][0,0,0],
    alpha = t2maps["alpha"][0,0,0],
    snr   = 20 * np.log10(t2maps["snr"][0,0,0]),
    sfr   = t2parts["sfr"][0,0,0],
    mfr   = t2parts["mfr"][0,0,0],
)
print(msg)
