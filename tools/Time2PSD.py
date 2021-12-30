"""
Created by:

@author: Elias Obreque
@Date: 12/16/2021 6:28 PM 
els.obrq@gmail.com

"""
import pandas as pd
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from datetime import datetime


def hann(n):
    return 0.5 * (1 - np.cos((2 * np.pi * np.arange(1, n + 1)) / (n + 1)))


def psdftt(y, nfft, fsamp, wndw, novlap):
    """
    Inputs:

    y: Time history vector to analyze
    nfft: Number of points in each ensemble, needs to be an even number,
    but does not have to be a power of two (2 ** x).
    fsamp: Sample rate of data, used to normalize output and generate frequency vector.
    wndw: A non-zero value will apply a Hanning window of length nfft to each
    ensemble (requires "hann" below).
    novlap: Number of points to overlap each ensemble, for example nfft=1024
    with novlap=512 is a 50% overlap
    REF:#: http://physionet.cps.unizar.es/~eduardo/docencia/tds/librohtml/welch1.htm

    Outputs:
    p: Power spectral density in units of [y units] squared per [fsamp units],
    for example g^2/Hz.
    f: Frequency vector.
    oarms: Overall rms value, square root of area under f-p curve
    """
    # Calculate number of available ensembles
    npts = len(y)
    ensembles = int(np.floor((npts - nfft) / (nfft - novlap)) + 1)

    # Initialize ensemble indexing variables
    n1 = 0
    n2 = int(nfft)
    dn = nfft - novlap

    # Initialize psd summation storage variable
    arg_sum = np.zeros(int(nfft))

    # Main program loop

    k = 0
    while k < ensembles:
        arg_y = y[int(n1): int(n2)]  # Extract current ensemble
        arg_y = arg_y - np.mean(arg_y)  # Remove mean

        if (wndw != 0):
            arg_y = arg_y * hann(nfft)  # Apply window if required

        arg_fft = np.fft.fft(arg_y, nfft)  # FFT of ensemble

        arg_abs = abs(arg_fft) ** 2  # Modulus squared of FFT
        arg_sum = arg_sum + arg_abs  # Accumulate in summation variable
        n1 = n1 + dn  # Increment ensemble index variables
        n2 = n2 + dn
        k += 1

    arg_sum = arg_sum / ensembles  # Average value of summed spectra

    # Compute window function normalization factor
    if (wndw != 0):
        wndw_sc = np.sum(hann(nfft) ** 2)
    else:
        wndw_sc = nfft

    # Power spectrum is symmetric about Nyquist frequency, use lower half and
    # multiply by 4. Then divide by 2 to convert peak^2 to rms^2. Thus scale
    # factor is 2.
    p = 2 * arg_sum[0: int(nfft / 2)]

    # Normalize to correct for window
    p = p / wndw_sc

    # Normalize spectral density for sampling units
    p = p / fsamp

    # Create frequency vector
    df = fsamp / nfft

    f = np.arange(0, fsamp / 2, df)

    # Calculate overall rms level from area under PSD curve
    oarms = np.sqrt(np.sum(p * df))
    return p, f, oarms


if __name__ == '__main__':
    sensors_name = ['cDAQ9189-1D36166Mod2/ai0',
                    'cDAQ9189-1D36166Mod3/ai1']
    # df = pd.read_csv("sensors/LogFile_2021-12-16-18-23-45.csv", sep=';')
    df = TdmsFile("./sensors/SinusoidalTest/LogFile_2021-12-29-19-11-31.tdms")

    # Channels
    acc_psd = (df['Log'][sensors_name[0]].data + df['Log'][sensors_name[1]].data) * 0.5
    # acc_psd[np.where(acc_psd > 20)] = 20.0
    time_psd = df['Log'][df['Log'].channels()[0].name].time_track()
    print(time_psd[1])
    fsamp = 6000

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

    plt.figure()
    plt.title('Fourier transform')
    plt.vlines(freq_fft[:int(nfft2 / 2)], 0, np.abs(acc_fft[:int(nfft2 / 2)]), colors='r')
    plt.grid()

    grms_acc = np.sqrt(np.mean(acc_psd ** 2))
    print(grms_acc)

    windows_ = -1
    if len_data < windows_:
        windows_ = len_data
    plt.figure(figsize=(10, 5))
    plt.title('Acceleration [g]')
    plt.step(time_psd[:nfft2], acc_psd[:nfft2], 'r')
    # plt.step(time_acc_real[:windows_], acc_psd_real[:windows_], 'b')
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Acc [g]")

    acc_fft, freq_fft, oarms_fft = psdftt(acc_psd, nfft2 / 8, fsamp, 0, nfft2 / 16)

    print("Grms [G]: ", oarms_fft)

    # PSD input: SpaceX
    HZ = [20, 100, 300, 700, 800, 925, 2000]
    GRMS2 = [0.0044, 0.0044, 0.01, 0.01, 0.03, 0.03, 0.00644]

    plt.figure(figsize=(10, 5))
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(freq_fft, acc_fft)
    # plt.plot(HZ, GRMS2, 'k')
    plt.grid(which='both', axis='both')
    plt.title('PSD: power spectral density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.ylim(1e-8, 1e1)
    plt.tight_layout()
    plt.show()

