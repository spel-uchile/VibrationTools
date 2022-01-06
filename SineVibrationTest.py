"""
Created by:

@author: Elias Obreque
@Date: 12/31/2021 12:46 AM 
els.obrq@gmail.com

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.Time2PSD import psdftt
from tools.acc2rdf import digitize

df = pd.read_csv('sine/spacex.txt', sep='\t', header=None)
saveData = True
M = 4096  # number of samples in the generated signal in time

HZ = df.values[:, 0]
ACC = df.values[:, 1]
slopes = np.diff(ACC) / np.diff(HZ)

fsamp = 600
pmin = 1 / max(HZ) / 4
if 1/pmin > fsamp:
    pmin = 1/fsamp

tmax = pmin * M
print('Period [sec]: ', tmax)

dt = 1 / fsamp
typeRate = 'Linear'  # Linear or Log
# if Log:
R = 4  # octave/minute
# if Linear:
R = 0.5  # frec / sec

n_octave = np.log(HZ[-1] / HZ[0]) / np.log(2)

freq_vec = [HZ[0]]
acc_vec = [0]
amp_vec = [ACC[0]]
t = 0
time = [t]
t += dt
while freq_vec[-1] <= HZ[-1]:
    if typeRate == 'Log':
        freq_vec = np.append(freq_vec, HZ[0] * (-1 + 2 ** (R / 60 * t)) / (R / 60 * t * np.log(2)))
    elif typeRate == 'Linear':
        freq_vec = np.append(freq_vec, 0.5 * R * t + HZ[0])
    for j in range(1, len(HZ)):
        if HZ[j - 1] <= freq_vec[-1] < HZ[j]:
            if abs(slopes[j - 1]) != np.inf:
                amp = ACC[j - 1] + slopes[j - 1] * (freq_vec[-1] - HZ[j - 1])
                break
            else:
                amp = ACC[j]
    acc_vec = np.append(acc_vec, np.sin(2 * np.pi * freq_vec[-1] * t))
    t += dt
    time.append(t)
    amp_vec.append(amp)

# Digitize
s_max = 2 ** 14 - 1   # Max level of 14-bit DAC is 16,383
s_zero = 2 ** 14 / 2  # "Zero" of waveform is 8,192
digitize(acc_vec, 0, '', s_max, s_zero)
# ================================================================================================
# PLOTS

plt.figure()
plt.title('freq')
plt.plot(time, freq_vec)
plt.grid()

plt.figure()
plt.title('Sine')
plt.xlabel('Time [min]')
plt.plot(np.array(time)/60, acc_vec)
plt.grid()

plt.figure()
plt.title('amp')
plt.plot(time, amp_vec)
plt.grid()


plt.figure()
plt.title('Sinusoidal vibration')
plt.plot(HZ, ACC)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration [g]')
plt.grid()

a, b, c = psdftt(acc_vec, len(acc_vec), fsamp, 0, len(acc_vec)/2)

plt.figure()
#plt.plot(b, a)
#plt.grid()

acc_ftt_ftt = np.fft.fft(acc_vec) / len(acc_vec)
freq_fft_fft = np.fft.fftfreq(len(acc_vec)) * fsamp

plt.figure()
plt.plot(freq_fft_fft[:int(len(acc_vec)/2)], acc_ftt_ftt[:int(len(acc_vec)/2)])
plt.grid()
plt.show()

