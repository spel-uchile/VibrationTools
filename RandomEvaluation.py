"""
Created by:

@author: Elias Obreque
@Date: 12/23/2021 12:20 PM 
els.obrq@gmail.com

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from tools.Time2PSD import psdftt
import bottleneck as bn

fsamp = 5000
signalDuration = 60.0
dP = 4096 / fsamp

# INPUT PSD
df_long = pd.read_csv("psd/ionDorbit_Longitudinal.txt", sep='\t', header=None)
df_lat = pd.read_csv("psd/ionDorbit_Lateral.txt", sep='\t', header=None)

HZ = [df_long.values[:, 0], df_lat.values[:, 0]]
GRMS2 = [df_long.values[:, 1], df_lat.values[:, 1]]

# INPUT signal
df_long = pd.read_csv("digitized/IONRandomLongitudinal", sep='\t', header=None)
df_lat = pd.read_csv("digitized/IONRandomLateral", sep='\t', header=None)

time_acc_expected = [np.zeros(len(df_long.values)),
                 np.zeros(len(df_lat.values))]

time_acc_expected[0] = df_long.values[:, 0]
time_acc_expected[1] = df_lat.values[:, 0]

acc_psd_expected = [np.zeros(len(time_acc_expected[0])), np.zeros(len(time_acc_expected[1]))]
acc_psd_expected[0] = df_long.values[:, 1]
acc_psd_expected[1] = df_lat.values[:, 1]

for i in range(2):
    graph_time = time_acc_expected[i]
    graph_acc = acc_psd_expected[i]
    while max(graph_time) < signalDuration:
        graph_time = np.append(graph_time, max(graph_time) + time_acc_expected[i] + 1/fsamp)
        graph_acc = np.append(graph_acc, acc_psd_expected[i])
    acc_psd_expected[i] = graph_acc
    time_acc_expected[i] = graph_time

len_data_expected = [len(acc_psd_expected[0]), len(acc_psd_expected[1])]

nfft2_expected = [2, 2]
for i in range(2):
    while nfft2_expected[i] < len_data_expected[i]:
        nfft2_expected[i] *= 2
    nfft2_expected[i] /= 2
    nfft2_expected[i] = int(nfft2_expected[i])

acc_fft_ref = [np.fft.fft(acc_psd_expected[0][:nfft2_expected[0]]) / nfft2_expected[0],
               np.fft.fft(acc_psd_expected[1][:nfft2_expected[1]]) / nfft2_expected[0]]  # Normalize
freq_fft_ref = [np.fft.fftfreq(nfft2_expected[0]) * fsamp,
                np.fft.fftfreq(nfft2_expected[1]) * fsamp]

# =============================================================================================#

# Sensor
df = [TdmsFile("./sensors/Suchai2/Longitudinal/LogFile_2021-12-24-14-41-46.tdms"),
      TdmsFile("./sensors/Suchai2/Lateral/LogFile_2021-12-24-14-36-27.tdms")]
name_signal = ['Longitudinal',
               'Lateral']

sensors_name = ['cDAQ9189-1D36166Mod2/ai0',
                'cDAQ9189-1D36166Mod3/ai1']
len_signals = len(df)

# Channels
acc_psd = []
time_psd = []
len_data = []
for dfi in df:
    acc_psd.append((dfi['Log'][sensors_name[0]].data + dfi['Log'][sensors_name[1]].data) * 0.5)
    # acc_psd[np.where(acc_psd > 20)] = 20.0
    time_psd.append(dfi['Log'][dfi['Log'].channels()[0].name].time_track())
    len_data.append(len(acc_psd[-1]))

nfft2 = [2] * len_signals
for i in range(len_signals):
    while nfft2[i] < len_data[i]:
        nfft2[i] *= 2
    nfft2[i] /= 2
    nfft2[i] = int(nfft2[i])

# Fourier transform
acc_fft = []
freq_fft = []
grms_acc = []
for i in range(len_signals):
    acc_fft.append(np.fft.fft(acc_psd[i][:nfft2[i]]) / nfft2[i])  # Normalize
    freq_fft.append(np.fft.fftfreq(nfft2[i]) * fsamp)
    grms_acc.append(np.sqrt(np.mean(acc_psd[i] ** 2)))

print('GRMS from acceleration: ', grms_acc)

psd_acc, psd_freq, oarms_fft = [], [], []
psd_acc_expected, psd_freq_expected, oarms_fft_expected = [], [], []
for i in range(2):
    acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd[i], nfft2[i] / 16, fsamp, 0, nfft2[i] / 32)
    psd_acc.append(acc_fft_)
    psd_freq.append(freq_fft_)
    oarms_fft.append(oarms_fft_)

    # acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd_expected[i], nfft2_expected[i] / 8, fsamp, 0, nfft2_expected[i] / 16)
    acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd_expected[i][:4096], 4096 / 8, fsamp, 0,
                                             4096 / 16)
    psd_acc_expected.append(acc_fft_)
    psd_freq_expected.append(freq_fft_)
    oarms_fft_expected.append(oarms_fft_)

print("Grms from measured PSD: ", oarms_fft)
print("Grms from expected PSD: ", oarms_fft_expected)

# =============================================================================================#

for i in range(2):
    plt.figure()
    plt.title('Fourier transform. ' + name_signal[i] + ' test')
    plt.vlines(freq_fft_ref[i][:int(nfft2_expected[i]/2)], 0, np.abs(acc_fft_ref[i][:int(nfft2_expected[i] / 2)]),
               color='k', label='Expected')
    plt.vlines(freq_fft[i][:int(nfft2[i] / 2)], 0, np.abs(acc_fft[i][:int(nfft2[i] / 2)]),
               color='b', label='Measured')
    plt.grid()
    plt.legend()

    windows_ = len(acc_psd[i])
    if windows_ > len(time_psd[i]):
        windows_ = len(time_psd[i])

    plt.figure(figsize=(10, 5))
    plt.title('Acceleration ' + name_signal[i])
    plt.step(time_psd[i][:windows_], acc_psd[i][:windows_], 'r')
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.step(time_acc_expected[i], acc_psd_expected[i], 'b')

    plt.figure(figsize=(10, 5))
    plt.title('PSD: power spectral density. ' + name_signal[i] + ' test - Grms = ' + str(oarms_fft[i]))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(psd_freq[i], psd_acc[i], label='Measured acceleration')
    plt.plot(bn.move_mean(psd_freq[i], window=20), bn.move_mean(psd_acc[i], window=20), label='Moving average of measured')
    plt.plot(HZ[i], GRMS2[i], 'k', label='PSD required')
    plt.plot(psd_freq_expected[i], psd_acc_expected[i], label='PSD expected')
    plt.grid(which='both', axis='both')
    plt.xlabel('Frequency')
    plt.ylabel('PSD [G^2/Hz]')
    plt.ylim(1e-8, 1e1)
    plt.tight_layout()
    plt.legend()

print('View plots')
plt.show()
