
#!/usr/bin/env python3
"""
touch_filter_plot.py
Quick visualization & filtering tool for touch CSV logs.

Usage:
    python touch_filter_plot.py <csv_file> --column Diff --filter moving_avg --window 5

Filters supported:
    - raw                : no filtering
    - moving_avg         : simple moving average (window)
    - median             : median filter (window)
    - ema                : exponential moving average (alpha)
    - butter             : low‑pass Butterworth (order, cutoff)

Requirements:
    pandas, numpy, matplotlib, scipy
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def exponential_moving_average(x, alpha):
    ema = np.zeros_like(x, dtype=float)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]
    return ema

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def apply_filter(values, ftype, args):
    if ftype == 'raw':
        return values
    elif ftype == 'moving_avg':
        window = int(args.window)
        return moving_average(values, window)
    elif ftype == 'median':
        window = int(args.window)
        if window % 2 == 0:
            window += 1  # need odd
        return medfilt(values, kernel_size=window)
    elif ftype == 'ema':
        return exponential_moving_average(values, float(args.alpha))
    elif ftype == 'butter':
        return butter_lowpass_filter(values, float(args.cutoff), float(args.fs), int(args.order))
    else:
        raise ValueError(f"Unsupported filter: {ftype}")

def main():
    parser = argparse.ArgumentParser(description='Touch CSV quick plot & filter')
    parser.add_argument('csv', help='CSV file path')
    parser.add_argument('--column', default='Diff', help='column to plot')
    parser.add_argument('--filter', default='raw', choices=['raw', 'moving_avg', 'median', 'ema', 'butter'])
    parser.add_argument('--window', type=int, default=5, help='window size for moving/median')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for EMA')
    parser.add_argument('--cutoff', type=float, default=5.0, help='cutoff freq for butter')
    parser.add_argument('--fs', type=float, default=60.0, help='sampling freq for butter')
    parser.add_argument('--order', type=int, default=3, help='order for butter')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        print(f"Column {args.column} not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    raw = df[args.column].to_numpy()
    filtered = apply_filter(raw, args.filter, args)

    plt.figure(figsize=(12,6))
    plt.plot(raw, label='raw', linewidth=0.7)
    if args.filter != 'raw':
        plt.plot(filtered, label=f'{args.filter}', linewidth=1.2)
    plt.title(f'{args.column} - {args.filter}')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel(args.column)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
