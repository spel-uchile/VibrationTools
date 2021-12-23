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


df_real = pd.read_csv("digitized/spacex_random_rigol", sep='\t')
time_acc_real = np.zeros(len(df_real.values) + 1)

time_acc_real[1:] = df_real.values[:, 0]

acc_psd_real = np.zeros(len(time_acc_real))
acc_psd_real[0] = df_real.columns[1]
acc_psd_real[1:] = df_real.values[:, 1]
len_data_real = len(acc_psd_real)

# df = pd.read_csv("sensors/LogFile_2021-12-16-18-23-45.csv", sep=';')
df = TdmsFile("./sensors/LogFile_2021-12-21-19-28-13.tdms")

# Channels
acc_psd = 0.5*(df.as_dataframe().values[:, 1] + df.as_dataframe().values[:, 0])
# acc_psd[np.where(acc_psd > 20)] = 20.0
time_psd = df['Log'][df['Log'].channels()[0].name].time_track()
fsamp = 5000

#time_psd = np.arange(0, 60, 1 / fsamp)
#amp = [1] #np.arange(20, 1800, 20)
#acc_psd = np.zeros(len(time_psd))
#for a in amp:
#    acc_psd += np.sin(2000 * 2 * np.pi * time_psd)

# acc_psd *= (5.13 / np.std(acc_psd))
len_data = len(acc_psd)
print(fsamp)
print(len_data)
nfft2 = 2
while nfft2 < len_data:
    nfft2 *= 2
nfft2 /= 2
nfft2 = int(nfft2)

acc_fft = np.fft.fft(acc_psd[:nfft2]) / nfft2    # Normalize
freq_fft = np.fft.fftfreq(nfft2) * fsamp
acc_fft_ref = np.fft.fft(acc_psd_real[:4096]) / 4096  # Normalize
freq_fft_ref = np.fft.fftfreq(4096) * fsamp

plt.figure()
plt.vlines(freq_fft_ref[:int(4096 / 2)], 0, np.abs(acc_fft_ref[:int(4096 / 2)]), colors='b')
plt.vlines(freq_fft[:int(nfft2 / 2)], 0, np.abs(acc_fft[:int(nfft2 / 2)]), colors='r')
plt.grid()

grms_acc = np.sqrt(np.mean(acc_psd ** 2))
print(grms_acc)

windows_ = -1
if len_data < windows_:
    windows_ = len_data
plt.figure(figsize=(20, 5))
plt.step(time_psd[:windows_], acc_psd[:windows_], 'r')
# plt.step(time_acc_real[:windows_], acc_psd_real[:windows_], 'b')
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Acc [g]")

acc_fft, freq_fft, oarms_fft = psdftt(acc_psd, nfft2 / 8, fsamp, 0, nfft2 / 16)

print("Grms [G]: ", oarms_fft)

# PSD input: SpaceX
HZ = [20, 100, 300, 700, 800, 925, 2000]
GRMS2 = [0.0044, 0.0044, 0.01, 0.01, 0.03, 0.03, 0.00644]

plt.figure(figsize=(15, 5))
plt.yscale('log')
plt.xscale('log')
plt.plot(freq_fft, acc_fft)
plt.plot(HZ, GRMS2, 'k')
plt.grid(which='both', axis='both')
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.ylim(1e-8, 1e1)
plt.tight_layout()
plt.show()