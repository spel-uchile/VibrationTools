
from tools.signal_utilities import EnterPSD
import matplotlib.pyplot as plt
import numpy as np
from tools.Time2PSD import psdftt
from scipy.interpolate import interp1d
from scipy.signal import welch, find_peaks, peak_prominences


def find_modal_peaks(x_freq, y_psd, freq_expected, psd_expected, max_freq=2000, distance=300, prominence=2.0, width=50):
    """
    Detecta picos modales evaluando la transmisibilidad (Q^2).
    """
    f_expected = interp1d(freq_expected, psd_expected, bounds_error=False, fill_value=np.nan)
    expected_interp = f_expected(x_freq)

    transmissibility = np.divide(y_psd, expected_interp,
                                 out=np.zeros_like(y_psd),
                                 where=(expected_interp != 0) & (~np.isnan(expected_interp)))
    peaks, properties = find_peaks(transmissibility, distance=distance, prominence=prominence, width=width)

    peaks = peaks[x_freq[peaks] <= max_freq]

    return peaks, properties


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

def print_peaks_error(freq, amp_freq):
    print('Modal frequency [Hz]')
    for i in range(len(freq)):
        print(round(freq[i], 1), 'Hz // ', round(amp_freq[i], 2), '%')
    return


def compare_peaks(prev_freqs, prev_amps, post_freqs, post_amps, sensor_name, axis_name, max_shift_hz=30):
    """
        Compares Previous and Post peaks and prints a Markdown table.
        """
    print(f"\n--- Modal Shift Summary: {sensor_name} ({axis_name}) ---")
    print(
        "| Mode | Prev. Freq. (Hz) | Post Freq. (Hz) | Freq. Shift (%) | Abs. Shift (Hz) | Prev. Amp. | Post Amp. | Amp. Shift (%) |")
    print(
        "|------|------------------|-----------------|-----------------|-----------------|------------|-----------|----------------|")
    modo = 1
    error_freq = []
    abs_error = []
    for pr_f, pr_a in zip(prev_freqs, prev_amps):
        if len(post_freqs) == 0:
            continue

        idx_closest = np.argmin(np.abs(np.array(post_freqs) - pr_f))
        po_f = post_freqs[idx_closest]
        po_a = post_amps[idx_closest]

        if abs(po_f - pr_f) < max_shift_hz:
            diff_f = po_f - pr_f
            pct_f = (diff_f / pr_f) * 100
            diff_a = po_a - pr_a
            pct_a = (diff_a / pr_a) * 100

            error_freq.append(diff_f)
            abs_error.append(pct_f)
            print(
                f"| {modo:4d} | {pr_f:19.1f} | {po_f:15.1f} | {pct_f:+15.2f}% | {diff_f:+14.1f} | {pr_a:13.4f} | {po_a:9.4f} | {pct_a:+14.2f}% |")
            modo += 1

    print_peaks_error(error_freq, abs_error)
    print("\n")


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

