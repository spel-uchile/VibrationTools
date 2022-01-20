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
from tools.math_tools import bisection


def get_sweep_sine(f1: float, f2: float, R: float, fsamp: float, typeRate: str = 'Log') -> object:
    def func(t2_):
        return f2 - f1 * (-1 + 2 ** (R * t2_)) / (R * t2_ * np.log(2))

    t2 = bisection(func, 1.0, 20.0, 500)

    pmin = 1 / f2 / 4
    if 1 / pmin > fsamp:
        pmin = 1 / fsamp

    print('Duration test [min]: ', t2)
    t2 *= 60
    dt = 1 / fsamp
    len_time = int(np.ceil(t2 / dt))

    time = np.arange(0, len_time) * dt
    freq_vec = np.zeros(len_time)
    freq_vec[0] = f1

    if typeRate == 'Log':
        freq_vec[1:] = [f1 * (-1 + 2 ** (R * time[i] / 60)) / (R * time[i] / 60 * np.log(2)) for i in
                        range(1, len_time)]
    elif typeRate == 'Linear':
        freq_vec = 0.5 * R * time + f1

    acc_vec = [np.sin(2 * np.pi * freq_vec[i] * time[i]) for i in range(len_time)]

    a, b, c = psdftt(acc_vec, int(len(acc_vec)/8), fsamp, 0, int(len(acc_vec) / 16))
    windows = min(len(a), len(b))

    acc_ftt_ftt = np.fft.fft(acc_vec) / len(acc_vec)
    freq_fft_fft = np.fft.fftfreq(len(acc_vec)) * fsamp

    return [acc_vec, freq_vec, time], [a, b, c, windows], [acc_ftt_ftt, freq_fft_fft]


if __name__ == '__main__':
    df = pd.read_csv('sine/resonance.txt', sep='\t', header=None)
    saveData = True

    HZ = df.values[:, 0]
    ACC = df.values[:, 1]
    slopes = np.diff(ACC) / np.diff(HZ)

    typeRate = 'Log'  # Linear or Log
    if typeRate == 'Log':
        R = 2  # octave/minute
    elif typeRate == 'Linear':
        R = 0.5  # frec / sec
    else:
        R = 4.0

    fsamp = 5000
    [acc_vec, freq_vec, time], [a, b, c, windows] = get_sweep_sine(HZ[0], HZ[1], R, fsamp, typeRate='Log')

    # Digitize
    s_max = 2 ** 14 - 1   # Max level of 14-bit DAC is 16,383
    s_zero = 2 ** 14 / 2  # "Zero" of waveform is 8,192
    # digitize(acc_vec, 0, '', s_max, s_zero)
    # ================================================================================================
    # PLOTS

    plt.figure()
    plt.plot(time, freq_vec)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.grid()

    plt.figure()
    plt.title('Sine')
    plt.xlabel('Time [min]')
    plt.ylabel('Acceleration [g]')
    plt.plot(np.array(time)/60, acc_vec)
    plt.grid()

    plt.figure()
    plt.title('Sinusoidal vibration')
    plt.plot(HZ, ACC)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Acceleration [g]')
    plt.grid()

    plt.figure()
    plt.title('PSD')
    plt.ylabel('PSD Grms [g^2/Hz]')
    plt.xlabel('Hz')
    plt.plot(b[:windows], a[:windows])
    plt.grid()

    acc_ftt_ftt = np.fft.fft(acc_vec) / len(acc_vec)
    freq_fft_fft = np.fft.fftfreq(len(acc_vec)) * fsamp

    plt.figure()
    plt.title('Fourier transform')
    plt.xlabel('Hz')
    plt.plot(freq_fft_fft[:int(len(acc_vec)/2)], np.abs(acc_ftt_ftt[:int(len(acc_vec)/2)]))
    plt.grid()
    plt.show()

