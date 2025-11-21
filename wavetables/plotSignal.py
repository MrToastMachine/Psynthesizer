#!/usr/bin/env python3
"""
Usage:
    python plot_waveform.py input.wav
"""

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def load_wav(path):
    """Load a WAV file and return (sample_rate, data_as_float32)."""
    sr, data = wavfile.read(path)

    # Normalize to float32 in range [-1, 1] if it's integer
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_int
    else:
        data = data.astype(np.float32)

    return sr, data


def plot_waveform(sr, data, title=None):
    """Plot waveform; supports mono or multi-channel."""
    num_samples = data.shape[0]
    t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)

    plt.figure(figsize=(10, 4))

    if data.ndim == 1:  # mono
        plt.plot(t, data, label="Mono")
    else:  # multi-channel (e.g. stereo)
        num_channels = data.shape[1]
        for ch in range(num_channels):
            plt.plot(t, data[:, ch], label=f"Channel {ch+1}", alpha=0.8)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title if title else "Waveform")
    plt.grid(True, alpha=0.3)
    if data.ndim > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the waveform of a WAV file."
    )
    parser.add_argument("wav_path", help="Path to the input .wav file")
    args = parser.parse_args()

    try:
        sr, data = load_wav(args.wav_path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.wav_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

    title = f"Waveform: {args.wav_path} (sr={sr} Hz)"
    plot_waveform(sr, data, title=title)


if __name__ == "__main__":
    main()

