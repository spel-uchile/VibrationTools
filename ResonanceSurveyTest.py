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
from SineVibrationTest import get_sweep_sine
from scipy.signal import find_peaks, peak_prominences
from tools.math_tools import print_peaks


fsamp = 5000

_, [a, b, c, windows], [acc_ftt_ftt, freq_fft_fft] = get_sweep_sine(5, 2000, 2, fsamp)

# Response signal
df = [TdmsFile("./sensors/Test-17-01-22/LogFile_2022-01-17-17-14-30-sweep1.tdms"),
      TdmsFile("./sensors/Test-17-01-22/LogFile_2022-01-17-17-21-23-sweep1a.tdms")]

name_signal = ['Previous-1',
               'Previous-2']
len_signals = len(name_signal)

sensors_name = ['cDAQ9189-1D36166Mod2/ai0',
                'cDAQ9189-1D36166Mod3/ai1']

# Channels
acc_psd = []
time_psd = []
len_data = []

i = 0
for dfi in df:
    acc_psd.append([])
    for sname in sensors_name:
        acc_psd[-1].append(dfi['Log'][sname].data)
    time_psd.append(dfi['Log'][dfi['Log'].channels()[0].name].time_track())
    len_data.append(min(len(time_psd[i]), len(acc_psd[i][-1])))
    i += 1

init_windows = [0, 0]

for i in range(len_signals):
    if len_data[i] % 2 != 0:
        len_data[i] -= 1
    acc_psd[i][0] = acc_psd[i][0][init_windows[i]:len_data[i]]
    acc_psd[i][0] -= np.mean(acc_psd[i][0])
    acc_psd[i][1] = acc_psd[i][1][init_windows[i]:len_data[i]]
    acc_psd[i][1] -= np.mean(acc_psd[i][1])
    time_psd[i] = time_psd[i][init_windows[i]:len_data[i]]

# Fourier transform
acc_fft = []
freq_fft = []
grms_acc = []
for i in range(len_signals):
    acc_fft.append([])
    grms_acc.append([])
    for k in range(len(sensors_name)):
        acc_fft[i].append(np.fft.fft(acc_psd[i][k][:len_data[i]]) / len_data[i])  # Normalize
        grms_acc[i].append(np.sqrt(np.mean(acc_psd[i][k] ** 2)))
    freq_fft.append(np.fft.fftfreq(len_data[i]) * fsamp)

# PSD
psd_acc, psd_freq, oarms_fft = [], [], []
for i in range(len(name_signal)):
    psd_acc.append([])
    psd_freq.append([])
    oarms_fft.append([])
    for k in range(len(sensors_name)):
        acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd[i][k], int(len_data[i] / 16), fsamp, 0, int(len_data[i] / 32))
        psd_acc[i].append(acc_fft_)
        psd_freq[i].append(freq_fft_)
        oarms_fft[i].append(oarms_fft_)

print("DATA Grms from measured PSD: ", oarms_fft)

# =====================================================================================================================
# PLOTS
data_mean = 100
prominence_level = 0.0005
prominence_width = 150
distance_peaks = 1000
color_s = ['b', 'k', 'o', 'g']

# for i in range(len(name_signal)):
#     fig_fft, axes_fft = plt.subplots(1, 2, figsize=(10, 5))
#     fig_fft.suptitle('Fourier transform. ' + name_signal[i] + ' test')
#     for k in range(len(sensors_name)):
#         axes_fft[k].plot(freq_fft_fft[:int(len(acc_ftt_ftt) / 2)], np.abs(acc_ftt_ftt[:int(len(acc_ftt_ftt) / 2)]),
#                          label='Expected', lw=0.7)
#
#         axes_fft[k].vlines(freq_fft[i][:int(len_data[i] / 2)], 0, np.abs(acc_fft[i][k][:int(len_data[i] / 2)]),
#                            color=color_s[k], label='Measured-ai'+str(k), lw=0.7)
#         peaks, properties = find_peaks(np.abs(acc_fft[i][k][:int(len_data[i] / 2)]),
#                                        distance=distance_peaks,
#                                        prominence=prominence_level,
#                                        width=prominence_width)
#         axes_fft[k].plot(freq_fft[i][peaks], np.abs(acc_fft[i][k][peaks]), 'r+', markersize=25)
#         axes_fft[k].grid()
#         axes_fft[k].legend()

# for i in range(len(name_signal)):
#     windows_ = max(len(acc_psd[i][0]), len(acc_psd[i][1]))
#     if windows_ > len(time_psd[i]):
#         windows_ = len(time_psd[i])
#
#     fig_acc, axes_acc = plt.subplots(1, 2, figsize=(10, 5))
#     fig_acc.suptitle('Acceleration ' + name_signal[i])
#     for k in range(len(sensors_name)):
#         axes_acc[k].step(time_psd[i][:windows_], acc_psd[i][k][:windows_], color_s[k], label='Measured-ai' + str(k),
#                          lw=0.7)
#         axes_acc[k].grid()
#         axes_acc[k].set_xlabel("Time [s]")
#         axes_acc[k].set_ylabel("Acceleration [g]")
#         axes_acc[k].legend()

for i in range(len(name_signal)):
    fig_psd, axes_psd = plt.subplots(2, 1, figsize=(10, 7))
    fig_psd.suptitle('PSD: power spectral density. ' + name_signal[i] + ' test - Grms = ' + str(oarms_fft[i]))
    for k in range(len(sensors_name)):
        max_wind = min(len(psd_freq[i][k]), len(psd_acc[i][k]))
        axes_psd[k].set_yscale('log')
        axes_psd[k].set_xscale('log')
        axes_psd[k].plot(b[:windows], a[:windows], label='Expected', lw=0.7)
        axes_psd[k].plot(psd_freq[i][k][:max_wind], psd_acc[i][k][:max_wind], label='Measured acceleration-ai'+str(k),
                         lw=0.7)

        x_mean = bn.move_mean(psd_freq[i][k][:max_wind], window=data_mean)
        y_mean = bn.move_mean(psd_acc[i][k][:max_wind], window=data_mean)
        peaks, properties = find_peaks(y_mean, distance=distance_peaks, prominence=prominence_level,
                                       width=prominence_width)

        prominences = peak_prominences(y_mean, peaks)
        contour_heights = y_mean[peaks] - prominences[0]

        axes_psd[k].plot(x_mean[peaks], y_mean[peaks], 'r+', markersize=25)
        axes_psd[k].plot(x_mean, y_mean,
                         label='Moving average of measured-ai'+str(k), lw=0.7)
        axes_psd[k].grid(which='both', axis='both')
        axes_psd[k].set_xlabel('Frequency')
        axes_psd[k].set_ylabel('PSD [G^2/Hz]')
        axes_psd[k].set_ylim(1e-11, 1)
        axes_psd[k].set_xlim(1, 4000)
        plt.tight_layout()
        axes_psd[k].legend()

# for i in range(len(name_signal)):
#     plt.figure()
#     plt.title('Histogram of ' + name_signal[i])
#     for k in range(len(sensors_name)):
#         plt.hist(acc_psd[i][k], bins=100, label='Hist measured-ai'+str(k), density=True)
#     plt.grid()
#     plt.legend()


fig_psd2, axes_psd2 = plt.subplots(2, 1, figsize=(10, 7))
fig_psd2.suptitle('PSD: power spectral density')
for k in range(len(sensors_name)):
    axes_psd2[k].set_yscale('log')
    axes_psd2[k].set_xscale('log')
    axes_psd2[k].grid(which='both', axis='both')
    axes_psd2[k].set_xlabel('Frequency')
    axes_psd2[k].set_ylabel('PSD [G^2/Hz]')
    axes_psd2[k].set_ylim(1e-11, 1)
    axes_psd2[k].set_xlim(1, 4000)
    plt.tight_layout()

    for i in range(len(name_signal)):
        max_wind = min(len(psd_freq[i][k]), len(psd_acc[i][k]))
        x_mean = bn.move_mean(psd_freq[i][k][:max_wind], window=data_mean)
        y_mean = bn.move_mean(psd_acc[i][k][:max_wind],  window=data_mean)
        peaks, properties = find_peaks(y_mean, distance=distance_peaks, prominence=prominence_level,
                                       width=prominence_width)
        print(name_signal[i] + ' - Measured acceleration-ai'+str(k))
        print_peaks(x_mean[peaks], y_mean[peaks])
        axes_psd2[k].plot(x_mean[peaks], y_mean[peaks], 'r+', markersize=25)
        axes_psd2[k].plot(psd_freq[i][k][:max_wind], psd_acc[i][k][:max_wind],
                          label=name_signal[i] + ' - Measured acceleration-ai'+str(k), lw=0.7)
        axes_psd2[k].plot(x_mean, y_mean,
                          label=name_signal[i] + ' - Moving average of measured-ai'+str(k), lw=0.7)
    axes_psd2[k].legend()
print('View plots')
plt.show()

