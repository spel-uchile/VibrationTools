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

fsamp = 6000
signalDuration = 60.0
dP = 4096 / fsamp

# INPUT PSD
df_long = pd.read_csv("psd/ionDorbit_Longitudinal_MPE.txt", sep='\t', header=None)
df_lat = pd.read_csv("psd/ionDorbit_Lateral_MPE.txt", sep='\t', header=None)

HZ = [df_long.values[:, 0], df_lat.values[:, 0]]
GRMS2 = [df_long.values[:, 1], df_lat.values[:, 1]]

# INPUT signal
df_long = pd.read_csv("digitized/IONRandomLongitudinalMPE6000Hz", sep='\t', header=None)
df_lat = pd.read_csv("digitized/IONRandomLateralMPE6000HZ", sep='\t', header=None)

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
df = [TdmsFile("./sensors/IONRandom2/LogFile_2022-01-03-16-45-46.tdms"),
      TdmsFile("./sensors/IONRandom2/LogFile_2022-01-03-16-48-12.tdms")]
name_signal = ['R: Lineal',
               'R: Log']

sensors_name = ['cDAQ9189-1D36166Mod2/ai0',
                'cDAQ9189-1D36166Mod3/ai1']
len_signals = len(df)

# Channels
acc_psd = []
time_psd = []
len_data = []
for dfi in df:
    acc_psd.append([])
    for sname in sensors_name:
        acc_psd[-1].append(dfi['Log'][sname].data)
    time_psd.append(dfi['Log'][dfi['Log'].channels()[0].name].time_track())
    len_data.append(len(time_psd[-1]))

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
    acc_fft.append([])
    grms_acc.append([])
    for k in range(len(sensors_name)):
        acc_fft[i].append(np.fft.fft(acc_psd[i][k][:nfft2[i]]) / nfft2[i])  # Normalize
        grms_acc[i].append(np.sqrt(np.mean(acc_psd[i][k] ** 2)))
    freq_fft.append(np.fft.fftfreq(nfft2[i]) * fsamp)

psd_acc, psd_freq, oarms_fft = [], [], []
psd_acc_expected, psd_freq_expected, oarms_fft_expected = [], [], []
for i in range(2):
    psd_acc.append([])
    psd_freq.append([])
    oarms_fft.append([])
    for k in range(len(sensors_name)):
        acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd[i][k], nfft2[i] / 16, fsamp, 0, nfft2[i] / 32)
        psd_acc[i].append(acc_fft_)
        psd_freq[i].append(freq_fft_)
        oarms_fft[i].append(oarms_fft_)

    # acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd_expected[i], nfft2_expected[i] / 8, fsamp, 0, nfft2_expected[i] / 16)
    acc_fft_, freq_fft_, oarms_fft_ = psdftt(acc_psd_expected[i][:4096], 4096 / 8, fsamp, 0,
                                             4096 / 16)
    psd_acc_expected.append(acc_fft_)
    psd_freq_expected.append(freq_fft_)
    oarms_fft_expected.append(oarms_fft_)

print("DATA Grms from measured PSD: ", oarms_fft)
print("Grms from expected PSD: ", oarms_fft_expected)

# =============================================================================================#
color_s = ['b', 'r']
for i in range(2):
    plt.figure()
    plt.title('Fourier transform. ' + name_signal[i] + ' test')
    #plt.vlines(freq_fft_ref[i][:int(nfft2_expected[i]/2)], 0, np.abs(acc_fft_ref[i][:int(nfft2_expected[i] / 2)]),
    #           color='k', label='Expected')
    for k in range(len(sensors_name)):
        plt.vlines(freq_fft[i][:int(nfft2[i] / 2)], 0, np.abs(acc_fft[i][k][:int(nfft2[i] / 2)]),
                   color=color_s[k], label='Measured-'+str(k))
    plt.grid()
    plt.legend()

    windows_ = max(len(acc_psd[i][0]), len(acc_psd[i][1]))
    if windows_ > len(time_psd[i]):
        windows_ = len(time_psd[i])

    plt.figure(figsize=(10, 5))
    plt.title('Acceleration ' + name_signal[i])
    for k in range(len(sensors_name)):
        plt.step(time_psd[i][:windows_], acc_psd[i][k][:windows_], color_s[k], label='Measured-' + str(k))
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    #plt.step(time_acc_expected[i], acc_psd_expected[i], 'k', label='Expected')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.title('PSD: power spectral density. ' + name_signal[i] + ' test - Grms = ' + str(oarms_fft[i]))
    plt.yscale('log')
    plt.xscale('log')
    for k in range(len(sensors_name)):
        plt.plot(psd_freq[i][k], psd_acc[i][k], label='Measured acceleration-'+str(k))
        plt.plot(bn.move_mean(psd_freq[i][k], window=20), bn.move_mean(psd_acc[i][k], window=20), label='Moving average of measured-'+str(k))
    #plt.plot(HZ[i], GRMS2[i], 'k', label='PSD required')
    #plt.plot(psd_freq_expected[i], psd_acc_expected[i], label='PSD expected')
    plt.grid(which='both', axis='both')
    plt.xlabel('Frequency')
    plt.ylabel('PSD [G^2/Hz]')
    plt.ylim(1e-8, 1e1)
    plt.tight_layout()
    plt.legend()

    plt.figure()
    plt.title('Histogram of ' + name_signal[i])
    for k in range(len(sensors_name)):
        plt.hist(acc_psd[i][k], bins=100, label='Hist measured-'+str(k), density=True)
    # plt.hist(acc_psd_expected[i], bins=100, label='Hist expected', density=True)
    plt.grid()
    plt.legend()

print('View plots')
plt.show()
