"""
Created by:

@author: Elias Obreque
@Date: 12/23/2021 12:47 AM 
els.obrq@gmail.com

"""

from tools.psd_syn import get_acc_from_psd
from tools.math_tools import get_psd_from_cos

M = 4096  # number of samples in the generated signal in time
s_max = 2 ** 14 - 1   # Max level of 14-bit DAC is 16,383
s_zero = 2 ** 14 / 2  # "Zero" of waveform is 8,192
fsamp = 5000  # Hz
signalDuration = 60

# Tom Irvine Method
# get_acc_from_psd(M, s_max, s_zero, fsamp, signalDuration)


# Cos method
get_psd_from_cos(M, s_max, s_zero, fsamp, signalDuration)
