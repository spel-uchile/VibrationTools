"""
Created by:

@author: Elias Obreque
@Date: 01/12/2022 10:23 AM
els.obrq@gmail.com

"""

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from tools.Time2PSD import psdftt
import bottleneck as bn

fsamp = 5000

# Response signal
df = TdmsFile("./sensors/Test-11-01-22/LogFile_2022-01-11-17-10-42-sweep-posterior-random-longitudinal.tdms")

sensors_name = ['cDAQ9189-1D36166Mod2/ai0',
                'cDAQ9189-1D36166Mod3/ai1']

# Channels
acc_psd = []

for sname in sensors_name:
    acc_psd.append(df['Log'][sname].data)
time_psd = df['Log'][df['Log'].channels()[0].name].time_track()
len_data = min(len(time_psd), len(acc_psd[0]))
if len_data % 2 != 0:
    len_data -= 1
acc_psd[0] = acc_psd[0][:len_data]
acc_psd[1] = acc_psd[1][:len_data]
time_psd = time_psd[:len_data]

# Fourier transform
acc_fft = []
grms_acc = []
freq_fft = []
for k in range(len(sensors_name)):
    acc_fft.append(np.fft.fft(acc_psd[k]))  # Normalize
    grms_acc.append(np.sqrt(np.mean(acc_psd[k] ** 2)))
    freq_fft.append(np.fft.fftfreq(len_data) * fsamp)

# PSD
psd_acc, psd_freq, oarms_fft = [], [], []
psd_acc_expected, psd_freq_expected, oarms_fft_expected = [], [], []
for k in range(len(sensors_name)):
    acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd[k], int(len_data / 16), fsamp, 0, int(len_data / 32))
    psd_acc.append(acc_fft_)
    psd_freq.append(freq_fft_)
    oarms_fft.append(oarms_fft_)

print("DATA Grms from measured PSD: ", oarms_fft)

# =====================================================================================================================
# PLOTS
color_s = ['b', 'r', 'o', 'g']
fig_fft, axes_fft = plt.subplots(1, 2, figsize=(10, 5))
fig_fft.suptitle('Fourier transform')
for k in range(len(sensors_name)):
    axes_fft[k].vlines(freq_fft[k][:int(len_data / 2)], 0, np.abs(acc_fft[k][:int(len_data / 2)]),
                       color=color_s[k], label='Me'
                                               'asured-' + str(k))
    axes_fft[k].grid()
    axes_fft[k].legend()

fig_acc, axes_acc = plt.subplots(1, 2, figsize=(10, 5))
fig_acc.suptitle('Acceleration')
for k in range(len(sensors_name)):
    axes_acc[k].step(time_psd, acc_psd[k], color_s[k], label='Measured-' + str(k))
    axes_acc[k].grid()
    axes_acc[k].set_xlabel("Time [s]")
    axes_acc[k].set_ylabel("Acceleration [g]")
    axes_acc[k].legend()

fig_psd, axes_psd = plt.subplots(2, 1, figsize=(10, 7))
fig_psd.suptitle('PSD: power spectral density - Grms = ' + str(oarms_fft))
for k in range(len(sensors_name)):
    max_wind = min(len(psd_freq[k]), len(psd_acc[k]))
    axes_psd[k].set_yscale('log')
    axes_psd[k].set_xscale('log')
    axes_psd[k].plot(psd_freq[k][:max_wind], psd_acc[k][:max_wind], label='Measured acceleration-'+str(k))
    axes_psd[k].plot(bn.move_mean(psd_freq[k][:max_wind], window=20), bn.move_mean(psd_acc[k][:max_wind], window=20),
                     label='Moving average of measured-'+str(k))
    axes_psd[k].grid(which='both', axis='both')
    axes_psd[k].set_xlabel('Frequency')
    axes_psd[k].set_ylabel('PSD [G^2/Hz]')
    axes_psd[k].set_ylim(1e-8, 1e1)
    plt.tight_layout()
    axes_psd[k].legend()

plt.figure()
plt.title('Histogram')
for k in range(len(sensors_name)):
    plt.hist(acc_psd[k], bins=100, label='Hist measured-'+str(k), density=True)
plt.grid()
plt.legend()
plt.show()
