#!/usr/bin/env python3
"""
Usage:
    python spectrogram.py input.wav
"""

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==========================
# Configurable parameters
# ==========================

# Matplotlib colormap name (e.g. 'viridis', 'magma', 'plasma', 'inferno', 'turbo')
COLORMAP = "magma"

# FFT window size and overlap for the spectrogram
NFFT = 1024        # Number of samples per FFT window
NOVERLAP = 512     # Overlap between windows


def load_wav_mono(path):
    """Load a WAV file and return (sample_rate, mono_signal_as_float32)."""
    sr, data = wavfile.read(path)

    # If stereo or multi-channel, convert to mono by averaging channels
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to float32 in range [-1, 1] if it's integer
    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_int
    else:
        data = data.astype(np.float32)

    return sr, data


def plot_spectrogram(sr, signal, title=None):
    """Plot a spectrogram for the given signal."""
    plt.figure(figsize=(10, 6))
    Pxx, freqs, bins, im = plt.specgram(
        signal,
        NFFT=NFFT,
        Fs=sr,
        noverlap=NOVERLAP,
        cmap=COLORMAP
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    if title:
        plt.title(title)
    else:
        plt.title("Spectrogram")

    cbar = plt.colorbar(im)
    cbar.set_label("Intensity [dB]")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot a spectrogram of a WAV file."
    )
    parser.add_argument("wav_path", help="Path to the input .wav file")
    args = parser.parse_args()

    try:
        sr, signal = load_wav_mono(args.wav_path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.wav_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

    title = f"Spectrogram: {args.wav_path} (sr={sr} Hz)"
    plot_spectrogram(sr, signal, title=title)


if __name__ == "__main__":
    main()

