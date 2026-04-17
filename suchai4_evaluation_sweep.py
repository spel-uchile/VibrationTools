"""
Created by: Elias Obreque
Adapted to exact original style for SUCHAI 4 Sweep/Resonance Survey
Report Generation Version
email: els.obrq@gmail.com
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import bottleneck as bn
from scipy.signal import welch
from SineVibrationTest import get_sweep_sine
from tools.math_tools import print_peaks, find_modal_peaks, compare_peaks

# =============================================================================
# CONFIGURATION
# =============================================================================
MAIN_DIR = 'data_suchai4_v2/'
output_dir = 'SUCHAI4_Plots'
os.makedirs(output_dir, exist_ok=True)

list_data = os.listdir(MAIN_DIR)
list_excel: list = [elem for elem in list_data if "xlsx" in elem]

axes_to_test = ['LATERAL_X', 'LATERAL_Y', 'LONG']
name_signal = ['Previous', 'Post']
len_signals = len(name_signal)
sensors_name = ['AI 2', 'AI 4']


data_mean = 10
prominence_level = 0.0005
prominence_width = 150
distance_peaks = 500
color_s = ['b', 'k', 'r', 'g']


def read_excel(file_path, pickle_filename):
    df1 = pd.read_excel(file_path, sheet_name='Data1', skiprows=[1])
    try:
        df2 = pd.read_excel(file_path, sheet_name='Data1-1')

        df1 = df1.drop(0)
        df2 = df2.drop(0)

        df_combined = pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        print(f"No Data2 sheet found. Using Data1 sheet.")
        df_combined = df1

    df_combined['Time'] = pd.to_numeric(df_combined['Time'])
    df_combined['AI 2'] = pd.to_numeric(df_combined['AI 2'])
    df_combined['AI 4'] = pd.to_numeric(df_combined['AI 4'])

    df_combined.to_pickle(pickle_filename)
    print(f"Data saved successfully to {pickle_filename}")
    return df_combined


for axis in axes_to_test:
    print(f"\n{'='*50}")
    print(f"ANALYZING AXIS: {axis}")
    print(f"{'='*50}")

    files_prev_post = [elem for elem in list_excel if axis in elem]
    df_prev = None
    df_post = None
    
    for name_of_excel_file in files_prev_post:
        print(f"\n--- Processing: {name_of_excel_file} ---")
        file_path = f'{MAIN_DIR}{name_of_excel_file}'
        pickle_filename = f'{MAIN_DIR}{name_of_excel_file[:-5]}.pkl'

        if os.path.exists(pickle_filename):
            print(f"Data already exists at {pickle_filename}. Loading pickle...")
            with open(pickle_filename, 'rb') as f:
                df_ = pd.read_pickle(f)
        else:
            print("Reading data from Excel...")
            df_ = read_excel(file_path, pickle_filename)

        if "prev" in name_of_excel_file.lower():
            df_prev = df_
        elif "post" in name_of_excel_file.lower():
            df_post = df_
        else:
            continue

    if df_prev is None or df_post is None:
        print("One or more dataframes are None. Skipping this axis.")
        continue

    dfs = [df_prev, df_post]

    dt = np.mean(np.diff(df_prev['Time']))
    fsamp = 1.0 / dt
    print("Sampling frequency: ", fsamp, "Hz")
    _, [a, b, c, windows], [acc_ftt_ftt, freq_fft_fft] = get_sweep_sine(5, 2000, 2, fsamp)

    a = np.array(a[:windows])
    b = np.array(b[:windows])

    valid_idx = b <= 2100
    a = a[valid_idx]
    b = b[valid_idx]

    windows = len(a)

    acc_psd = []
    time_psd = []
    len_data = []

    for i, dfi in enumerate(dfs):
        acc_psd.append([])
        for sname in sensors_name:
            acc_psd[-1].append(dfi[sname].values)
        time_psd.append(dfi['Time'].values)
        len_data.append(min(len(time_psd[i]), len(acc_psd[i][-1])))

    init_windows = [0, 0]

    for i in range(len_signals):
        if len_data[i] % 2 != 0:
            len_data[i] -= 1
        acc_psd[i][0] = acc_psd[i][0][init_windows[i]:len_data[i]]
        acc_psd[i][0] -= np.mean(acc_psd[i][0])
        acc_psd[i][1] = acc_psd[i][1][init_windows[i]:len_data[i]]
        acc_psd[i][1] -= np.mean(acc_psd[i][1])
        time_psd[i] = time_psd[i][init_windows[i]:len_data[i]]

    psd_acc, psd_freq, oarms_fft = [], [], []
    for i in range(len(name_signal)):
        psd_acc.append([])
        psd_freq.append([])
        oarms_fft.append([])
        for k in range(len(sensors_name)):
            nperseg_val = int(len_data[i] / 16)
            noverlap_val = int(len_data[i] / 32)

            f, Pxx = welch(acc_psd[i][k], fs=fsamp, nperseg=nperseg_val, noverlap=noverlap_val)
            psd_acc[i].append(Pxx)
            psd_freq[i].append(f)

            grms = np.sqrt(np.trapezoid(Pxx, f))
            oarms_fft[i].append(round(grms, 4))

    # 1. ACCELERATION PLOT
    for i in range(len(name_signal)):
        windows_ = max(len(acc_psd[i][0]), len(acc_psd[i][1]))
        if windows_ > len(time_psd[i]):
            windows_ = len(time_psd[i])

        fig_acc, axes_acc = plt.subplots(1, 2, figsize=(10, 5))
        fig_acc.suptitle(f'Acceleration {name_signal[i]} - {axis}')
        for k in range(len(sensors_name)):
            axes_acc[k].step(time_psd[i][:windows_], acc_psd[i][k][:windows_], color_s[k], label='Measured-A' + str(k *2 + 2), lw=0.7)
            axes_acc[k].grid()
            axes_acc[k].set_xlabel("Time [s]")
            axes_acc[k].set_ylabel("Acceleration [g]")
            axes_acc[k].legend()
        fig_acc.savefig(f"{output_dir}/1_Time_{axis}_{name_signal[i]}.png", dpi=300, bbox_inches='tight')

    # 2. INDIVIDUAL PSD PLOT
    for i in range(len(name_signal)):
        fig_psd, axes_psd = plt.subplots(2, 1, figsize=(10, 7))
        oarms_text = ""
        for k in range(len(sensors_name)):
            oarms_text += f"A{k *2 + 2}: " + str(oarms_fft[i][k]) + ", "
            max_wind = min(len(psd_freq[i][k]), len(psd_acc[i][k]))
            axes_psd[k].set_yscale('log')
            axes_psd[k].set_xscale('log')
            axes_psd[k].plot(b[:windows], a[:windows], label='Expected', lw=0.7)
            axes_psd[k].plot(psd_freq[i][k][:max_wind], psd_acc[i][k][:max_wind], label='Measured acceleration-A'+str(k *2 + 2), lw=0.7)

            x_mean = bn.move_mean(psd_freq[i][k][:max_wind], window=data_mean)[data_mean-1:]
            y_mean = bn.move_mean(psd_acc[i][k][:max_wind], window=data_mean)[data_mean-1:]

            peaks, _ = find_modal_peaks(x_mean, y_mean, b, a, max_freq=2000, distance=distance_peaks,
                                        prominence=prominence_level, width=prominence_width)

            axes_psd[k].plot(x_mean[peaks], y_mean[peaks], 'r+', markersize=25)
            axes_psd[k].vlines(x_mean[peaks], 0, 1, 'k', linestyles='dashed', lw=0.6)
            axes_psd[k].plot(x_mean, y_mean, label='Moving average of measured-A' + str(k *2 + 2), lw=0.7)

            for j in range(len(x_mean[peaks])):
                axes_psd[k].text(x_mean[peaks][j], y_mean[peaks][j] + 0.0015 * (1 - (-1)**j),
                                 str(round(x_mean[peaks][j], 1)) + " Hz", horizontalalignment='left',
                                 fontsize=8, verticalalignment='bottom', rotation=70, weight='bold')
            axes_psd[k].grid(which='both', axis='both')
            axes_psd[k].set_xlabel('Frequency')
            axes_psd[k].set_ylabel('PSD [G^2/Hz]')
            axes_psd[k].set_xlim(1, 4000)
            axes_psd[k].set_ylim(1e-10, 10)
            axes_psd[k].legend(loc='lower left')

        if oarms_text.endswith(", "):
            oarms_text = oarms_text[:-2]
        fig_psd.suptitle(f'PSD: power spectral density. {name_signal[i]} test - {axis} - Grms = {oarms_text}')
        plt.tight_layout()
        fig_psd.savefig(f"{output_dir}/2_PSD_{axis}_{name_signal[i]}.png", dpi=300, bbox_inches='tight')

    # 3. PSD PLOT OVERLAY
    fig_psd2, axes_psd2 = plt.subplots(2, 1, figsize=(10, 7))
    fig_psd2.suptitle(f'PSD: power spectral density - {axis}')
    peak_data = {k: {'prev_f': [], 'prev_a': [], 'post_f': [], 'post_a': []} for k in range(len(sensors_name))}

    for k in range(len(sensors_name)):
        axes_psd2[k].set_yscale('log')
        axes_psd2[k].set_xscale('log')
        axes_psd2[k].grid(which='both', axis='both')
        axes_psd2[k].set_xlabel('Frequency')
        axes_psd2[k].set_ylabel('PSD [G^2/Hz]')
        axes_psd2[k].set_ylim(1e-10, 10)
        axes_psd2[k].set_xlim(1, 4000)


        for i in range(len(name_signal)):
            max_wind = min(len(psd_freq[i][k]), len(psd_acc[i][k]))
            x_mean = bn.move_mean(psd_freq[i][k][:max_wind], window=data_mean)[data_mean-1:]
            y_mean = bn.move_mean(psd_acc[i][k][:max_wind], window=data_mean)[data_mean-1:]

            peaks, _ = find_modal_peaks(x_mean, y_mean, b, a, max_freq=2000, distance=distance_peaks,
                                        prominence=prominence_level, width=prominence_width)

            if name_signal[i] == 'Previous':
                peak_data[k]['prev_f'] = x_mean[peaks]
                peak_data[k]['prev_a'] = y_mean[peaks]
            else:
                peak_data[k]['post_f'] = x_mean[peaks]
                peak_data[k]['post_a'] = y_mean[peaks]

            print(f'{name_signal[i]} - Measured acceleration-A{k *2 + 2} ({axis})')
            print_peaks(x_mean[peaks], y_mean[peaks])

            axes_psd2[k].plot(x_mean[peaks], y_mean[peaks], 'r+', markersize=25)
            axes_psd2[k].plot(psd_freq[i][k][:max_wind], psd_acc[i][k][:max_wind],
                              label=name_signal[i] + ' - Measured acceleration-A'+str(k *2 + 2), lw=0.7)
            axes_psd2[k].plot(x_mean, y_mean,
                              label=name_signal[i] + ' - Moving average of measured-A'+str(k *2 + 2), lw=0.7)

        axes_psd2[k].legend()

        compare_peaks(peak_data[k]['prev_f'], peak_data[k]['prev_a'],
                      peak_data[k]['post_f'], peak_data[k]['post_a'],
                      f"A{k * 2 + 2}", axis)

    plt.tight_layout()
    fig_psd2.savefig(f"{output_dir}/3_Overlay_{axis}_{name_of_excel_file}.png", dpi=300, bbox_inches='tight')

    print(f'\nAll plots successfully saved in /{output_dir}/')
    plt.show()