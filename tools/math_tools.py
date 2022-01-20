
from tools.signal_utilities import EnterPSD
import matplotlib.pyplot as plt
import numpy as np
from tools.Time2PSD import psdftt


def bisection(f, a, b, N):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    f = lambda x: x**2 - x - 1
    bisection(f,1,2,25)
    1.618033990263939
    f = lambda x: (2*x - 1)*(x - 3)
    bisection(f,0,1,10)
    0.5
    '''

    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1, N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2


def print_peaks(freq, peaks_amp):
    print('Modal frequency [Hz]')
    for i in range(len(freq)):
        print(round(freq[i], 1), 'Hz // ', round(peaks_amp[i], 4), 'g^2/Hz')

    return


def get_psd_from_cos(M, s_max, s_zero, sr, signalDuration):
    tpi = 2 * np.pi

    freq_spec, amp_spec, rms, num, slope = EnterPSD()

    nm1 = num - 1
    LS = nm1

    three_rms = 3 * rms

    # print(" ")
    # print(" Enter duration(sec)")
    # tmax = enter_float()
    pmin = 1 / max(freq_spec) / 4
    if 1 / pmin > sr:
        pmin = 1 / sr

    tmax = pmin * M
    print('Period [sec]: ', tmax)
    fmax = max(freq_spec)

    dt = 1 / sr

    npd = int(np.ceil(tmax / dt))

    num_fft = 2

    while num_fft < npd:
        num_fft *= 2

    N = num_fft
    df = 1. / (N * dt)

    m2 = int(num_fft / 2)

    fft_freq = np.linspace(0, (m2 - 1) * df, m2)
    fft_freq2 = np.linspace(0, (num_fft - 1) * df, num_fft)

    print(" Interpolate specification")

    if fft_freq[0] <= 0:
        fft_freq[0] = 0.5 * fft_freq[1]

    x = np.log10(fft_freq)
    xp = np.log10(freq_spec)
    yp = np.log10(amp_spec)

    y = np.interp(x, xp, yp, left=-10, right=-10)

    sq_spec = np.sqrt(10 ** y)

    time = np.linspace(0, tmax, M)

    xki = np.zeros(M)
    for i in range(m2):
        xki += (10*sq_spec[i]**2) * np.cos(tpi * fft_freq[i] * time)

    a, b, c = psdftt(xki, M/2, sr, 0, M/4)
    windows = min(len(a), len(b))

    print('Grms:', c)
    plt.figure()
    plt.title('PSD')
    plt.ylabel('PSD Grms [g^2/Hz]')
    plt.xlabel('Hz')
    plt.plot(b[:windows], a[:windows], 'o-', label='PSD')
    plt.plot(freq_spec, amp_spec, label='Expected')
    plt.plot(fft_freq, 10 ** y, label='Interpolation')
    plt.xscale('Log')
    plt.yscale('Log')
    plt.grid()
    plt.legend()
    plt.show()
    return

