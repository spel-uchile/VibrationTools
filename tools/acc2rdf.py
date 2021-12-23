"""
Created by:

@author: Elias Obreque
@Date: 12/20/2021 1:57 PM 
els.obrq@gmail.com

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def testing_wave(filename, freq, fsamp):
    signalDuration = 10
    print(signalDuration)
    dt = 1 / fsamp
    nd = signalDuration / dt

    nfft = []
    for ind in nd:
        nfft_tmp = 2
        while nfft_tmp < ind:
            nfft_tmp *= 2
        nfft.append(nfft_tmp)

    print(nfft * dt)
    t = [np.arange(0, nfft[i] * dt[i], dt[i]) for i in range(len(fsamp))]
    acc_random = [np.sin(2 * np.pi * freq * it) + np.sin(2 * np.pi * freq * 0.5 * it) for it in t]

    plt.figure()
    [plt.plot(t[i], acc_random[i], '.-', label=str(fsamp[i]) + ' Hz') for i in range(len(fsamp))]
    plt.legend()
    plt.grid()

    # Create a "fancy" function for the arbitrary waveform generator
    s_max = 2**14 - 1   # Max level of 14-bit DAC is 16,383
    s_zero = 2**14 / 2  # "Zero" of waveform is 8,192

    randomAccel_frame = [np.ceil((s_max / 2.0) * np.array(iacc_random) / max(iacc_random) + (s_max / 2.0)) for iacc_random in acc_random]

    randomAccel_frame = [irandomAccel_frame.astype(np.uint16) for irandomAccel_frame in randomAccel_frame] # Create a 16-bit integer
    acc_frame = [randomAccel_frame[i] - np.mean(randomAccel_frame[i]) for i in range(len(fsamp))]

    randomAccel_frame_ = [randomAccel_frame[i].astype(np.uint16) for i in range(len(fsamp))]   # Create a 16-bit integer
    filename += ".rdf"
    randomAccel_frame_[0].astype('int16').tofile(filename)

    random_fft = [np.fft.fft(acc_frame[i]) / nfft[i] for i in range(len(fsamp))]
    freq_fft = [np.fft.fftfreq(nfft[i]) * fsamp[i] for i in range(len(fsamp))]

    # FFT
    plt.figure()
    [plt.plot(freq_fft[i], abs(random_fft[i]), label=str(fsamp[i]) + ' Hz') for i in range(len(fsamp))]
    plt.legend()
    plt.grid()

    # RDF
    plt.figure()
    [plt.step(t[i], randomAccel_frame[i], label=str(fsamp[i]) + ' Hz') for i in range(len(fsamp))]
    plt.legend()
    plt.grid()
    plt.show()


def acc_random2rdf(filename):
    df = pd.read_csv(filename, sep="\t")
    time_array = df.values[:, 0]
    acc_random = df.values[:, 1]
    acc_random /= np.max(abs(acc_random))
    print(len(acc_random))
    # Create a "fancy" function for the arbitrary waveform generator
    s_max = 2 ** 14 - 1  # Max level of 14-bit DAC is 16,383
    s_zero = 2 ** 14 / 2  # "Zero" of waveform is 8,192

    randomAccel_frame = np.ceil((s_max / 2.0) * np.array(acc_random) + (s_max / 2.0))

    randomAccel_frame = randomAccel_frame.astype(np.uint16)  # Create a 16-bit integer
    filename += ".rdf"
    randomAccel_frame.astype('int16').tofile(filename)


def digitize(psd_Accel, n, filename, s_max, s_zero):
    randomAccel = np.copy(psd_Accel)
    randomAccel /= max(abs(randomAccel))
    randomAccel_frame = np.ceil((s_max / 2.0) * np.array(randomAccel) + (s_max / 2.0))
    randomAccel_frame = randomAccel_frame.astype(np.uint16)  # Create a 16-bit integer
    plt.figure(n)
    plt.title("Digitized acceleration")
    plt.plot(randomAccel_frame, linewidth=1.0)
    plt.grid()
    plt.ylabel("Binary level")
    plt.xlabel("Number of data")
    randomAccel_frame.astype('int16').tofile(filename)


def testing_rigol():
    M = 4096
    t = np.linspace(0, 1, M)
    a_20 = np.cos(2 * np.pi * 20 * t)
    a_100 = np.cos(2 * np.pi * 100 * t)
    a_1000 = np.cos(2 * np.pi * 1000 * t)
    a_2000 = np.cos(2 * np.pi * 2000 * t)

    plt.figure()
    plt.plot(a_100)
    plt.show()

    digitize(a_20, "test20Hz.rdf")
    digitize(a_100, "test100Hz.rdf")
    digitize(a_100, "test1000Hz.rdf")
    digitize(a_2000, "test2000Hz.rdf")
    plt.show()


if __name__ == '__main__':
    filename = 'sin_wave'
    freq = 1
    fsamp = np.array([6000])
    # testing_wave(filename, freq, fsamp)

    # testing_rigol()

    filename = 'spacex_random_rigol'
    acc_random2rdf(filename)
