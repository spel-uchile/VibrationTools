########################################################################
# program: psdint.py
# author: Tom Irvine
# version: 1.3
# date: September 13, 2013
# description:
#
#  Calculate the overall RMS level for a power spectral density
#
########################################################################

from __future__ import print_function

from signal_utilities import EnterPSD

import matplotlib.pyplot as plt

########################################################################

a, b, rms, num, slope = EnterPSD()

########################################################################

print(" ")
print(" view plot ")

plt.figure(1)
plt.plot(a, b)
title_string = 'Power Spectral Density   ' + str("%6.3g" % rms) + ' GRMS Overall '
plt.title(title_string)
plt.ylabel(' Accel (G^2/Hz)')
plt.xlabel(' Frequency (Hz) ')
plt.grid(which='both')
plt.savefig('power_spectral_density')
plt.xscale('log')
plt.yscale('log')
plt.show()