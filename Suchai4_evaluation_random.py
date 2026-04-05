"""
Created by Elias Obreque
Date: 29/03/2026
email: els.obrq@gmail.com
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import bottleneck as bn
from scipy.signal import welch

list_data = os.listdir('data2026_suchai4')
list_excel = [elem for elem in list_data if "xlsx" in elem]

# INPUT PSD
df_long = pd.read_csv("psd/ionDorbit_Longitudinal_PFM.txt", sep='\t', header=None)
df_lat = pd.read_csv("psd/ionDorbit_Lateral_PFM.txt", sep='\t', header=None)

# Parameters for Quasi-static analysis
winn = 20
delta_freq_factor_crit = 20
nperseg = 4096 # Window size for PSD calculation

for name_of_excel_file in list_excel:
    print(f"\n--- Processing: {name_of_excel_file} ---")
    file_path = f'data2026_suchai4/{name_of_excel_file}'
    pickle_filename = f'data2026_suchai4/{name_of_excel_file[:-5]}.pkl'

    if os.path.exists(pickle_filename):
        print(f"Data already exists at {pickle_filename}. Loading pickle...")
        with open(pickle_filename, 'rb') as f:
            df_combined = pd.read_pickle(f)
    else:
        print("Reading data from Excel...")
        df1 = pd.read_excel(file_path, sheet_name='Data1')
        df2 = pd.read_excel(file_path, sheet_name='Data1-1')

        df1 = df1.drop(0)
        df2 = df2.drop(0)

        df_combined = pd.concat([df1, df2], ignore_index=True)

        df_combined['Time'] = pd.to_numeric(df_combined['Time'])
        df_combined['AI 2'] = pd.to_numeric(df_combined['AI 2'])
        df_combined['AI 4'] = pd.to_numeric(df_combined['AI 4'])

        df_combined.to_pickle(pickle_filename)
        print(f"Data saved successfully to {pickle_filename}")

    # =========================================================================
    # 1. SIGNAL PROCESSING & PSD CALCULATION
    # =========================================================================
    # Calculate actual sampling frequency from the Time vector
    dt = np.mean(np.diff(df_combined['Time']))
    fsamp = 1.0 / dt

    sensors = ['AI 2', 'AI 4']
    colors = ['blue', 'red']

    psd_freqs = []
    psd_accs = []
    oarms_fft = []
    freq_qs = []
    g_acc = []

    print(f"Calculating PSD and Quasi-static equivalents (fs = {fsamp:.1f} Hz)...")

    for sensor in sensors:
        signal = df_combined[sensor].values

        # Calculate PSD using standard Welch's method
        f, Pxx = welch(signal, fs=fsamp, nperseg=nperseg)
        psd_freqs.append(f)
        psd_accs.append(Pxx)

        # Calculate Overall RMS (Grms)
        grms = np.sqrt(np.trapz(Pxx, f))
        oarms_fft.append(grms)

        # Moving Average for Quasi-static
        max_wind = len(f)
        x_temp = bn.move_mean(f, window=winn)[winn-1:]
        y_temp = bn.move_mean(Pxx, window=winn)[winn-1:]

        # Peak analysis
        i_20 = np.argmin(abs(x_temp - 10)) # Look after 10Hz
        if len(y_temp[i_20:]) > 0:
            i_max = np.argmax(y_temp[i_20:])
            peak_freq = x_temp[i_max + i_20]

            factor_crit = np.argmin(abs(x_temp - peak_freq - delta_freq_factor_crit))
            x_area = x_temp[i_20:factor_crit]
            y_area = y_temp[i_20:factor_crit]

            # Accumulated G
            g_accumulated = np.sqrt(np.sum(y_area * np.mean(np.diff(x_area))))
        else:
            peak_freq = 0
            g_accumulated = 0

        freq_qs.append(peak_freq)
        g_acc.append(g_accumulated)

        print(f"  {sensor}: Grms = {grms:.3f}, QS Freq = {peak_freq:.2f} Hz, Gacc = {g_accumulated:.3f}, 3Sigma = {g_accumulated*3:.3f}")

    # =========================================================================
    # 2. PLOTTING
    # =========================================================================

    # --- Plot A: Time Domain ---
    plt.figure(figsize=(12, 4))
    plt.title(f'Time History - {name_of_excel_file}')
    for i, sensor in enumerate(sensors):
        plt.plot(df_combined['Time'], df_combined[sensor], label=f'{sensor}', color=colors[i], linewidth=0.5, alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude (g)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Plot B: Power Spectral Density (PSD) ---
    fig_psd, axes_psd = plt.subplots(2, 1, figsize=(12, 8))
    fig_psd.suptitle(f'PSD: Power Spectral Density - {name_of_excel_file}')

    for i, sensor in enumerate(sensors):
        axes_psd[i].set_yscale('log')
        axes_psd[i].set_xscale('log')

        # Plot raw PSD and Moving Average
        axes_psd[i].plot(psd_freqs[i], psd_accs[i], color=colors[i], label=f'{sensor} Measured (Grms={oarms_fft[i]:.2f})', alpha=0.5)

        x_temp = bn.move_mean(psd_freqs[i], window=winn)[winn - 1:]
        y_temp = bn.move_mean(psd_accs[i], window=winn)[winn - 1:]
        axes_psd[i].plot(x_temp, y_temp, color='black', label='Moving average')

        # NOTE: Add your reference HZ and GRMS2 profiles here if you have them!
        # axes_psd[i].plot(HZ[i], GRMS2[i], 'k--', label='PSD required')

        axes_psd[i].grid(which='both', axis='both')
        axes_psd[i].set_xlabel('Frequency (Hz)')
        axes_psd[i].set_ylabel('PSD [g^2/Hz]')
        axes_psd[i].set_ylim(1e-6, 1e1)
        axes_psd[i].set_xlim(10, fsamp/2) # Nyquist limit
        axes_psd[i].legend()
    plt.tight_layout()

    # --- Plot C: Quasi-Static Analysis ---
    fig_qs, axes_qs = plt.subplots(2, 1, figsize=(12, 8))
    fig_qs.suptitle(f'Quasi-Static Equivalent - {name_of_excel_file}')

    for i, sensor in enumerate(sensors):
        axes_qs[i].set_yscale('log')
        axes_qs[i].set_xscale('log')

        axes_qs[i].plot(psd_freqs[i], psd_accs[i], label=f'{sensor} Measured', color=colors[i], lw=0.2, alpha=0.5)

        x_temp = bn.move_mean(psd_freqs[i], window=winn)[winn-1:]
        y_temp = bn.move_mean(psd_accs[i], window=winn)[winn-1:]
        axes_qs[i].plot(x_temp, y_temp, color='orange', label=f'Moving average (Gacc={g_acc[i]:.2f})')

        # Fill area and draw vertical line for critical peak
        i_20 = np.argmin(abs(x_temp - 10))
        peak_idx = np.argmin(abs(x_temp - freq_qs[i]))
        factor_crit = np.argmin(abs(x_temp - freq_qs[i] - delta_freq_factor_crit))

        axes_qs[i].fill_between(x_temp[i_20:factor_crit], y_temp[i_20:factor_crit], color='orange', alpha=0.3)
        axes_qs[i].vlines(freq_qs[i], 0, 10, color='red', linestyles='dashed', lw=1.5)
        axes_qs[i].text(freq_qs[i], 1e-6, f"{freq_qs[i]:.1f} Hz", weight='bold', color='red', horizontalalignment='left')

        axes_qs[i].grid(which='both', axis='both')
        axes_qs[i].set_xlabel('Frequency (Hz)')
        axes_qs[i].set_ylabel('PSD [g^2/Hz]')
        axes_qs[i].set_ylim(1e-7, 1e1)
        axes_qs[i].set_xlim(10, fsamp/2)
        axes_qs[i].legend()
    plt.tight_layout()

    plt.show()