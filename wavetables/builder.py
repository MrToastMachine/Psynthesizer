#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D
import wave
import os


# ===================== Parameters ===================== #
SAMPLE_RATE = 48000           # Hz, for audio export
TABLE_SIZE = 512              # Samples per wavetable frame (one cycle)
NUM_TABLES = 200              # Number of frames over time
BASE_FREQ = 220.0             # Hz, only conceptually used (one cycle per table)
DRIVE = 3                   # Sine drive into the folder (higher = more folding)
THRESH_START = 0            # High threshold -> almost pure sine
THRESH_END = -2            # Low threshold -> heavy folding
OUT_FILE = "wavefold_wavetable.wav"


# ===================== Core DSP ===================== #
def wavefolder(x: np.ndarray, threshold: float, iterations: int = 10) -> np.ndarray:
    """
    Simple wavefolder:
    - Anything above |threshold| is reflected back toward 0.
    - Iterate a few times to handle multiple folds.
    """
    y = x.copy()
    t = abs(threshold)

    for _ in range(iterations):
        over = np.abs(y) > t
        if not np.any(over):
            break
        # Reflect the part beyond the threshold back inward
        y[over] = np.sign(y[over]) * (2 * t - np.abs(y[over]))

    return y


def build_wavetable(num_tables=NUM_TABLES,
                    table_size=TABLE_SIZE,
                    drive=DRIVE,
                    thresh_start=THRESH_START,
                    thresh_end=THRESH_END):
    """
    Build a 2D wavetable:
    - axis 0: 'time' (table index)
    - axis 1: phase within one cycle
    """
    phases = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
    base_cycle = np.sin(phases)

    thresholds = np.logspace(thresh_start, thresh_end, num_tables)
    tables = np.zeros((num_tables, table_size), dtype=np.float32)

    for i, th in enumerate(thresholds):
        folded = wavefolder(base_cycle, th)
        tables[i, :] = folded

    return tables, phases, thresholds


# ===================== Visualization ===================== #
def plot_wavetable_3d(tables: np.ndarray, phases: np.ndarray):
    """
    Show the wavetable as a 3D surface:
      X: phase (radians)
      Y: table index (time)
      Z: amplitude
    """
    num_tables, table_size = tables.shape

    # Meshgrid: (num_tables, table_size)
    T_indices = np.arange(num_tables)
    T, P = np.meshgrid(T_indices, phases, indexing="ij")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(P, T, tables, rstride=2, cstride=4,
                           linewidth=0, antialiased=True)

    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Table index (time)")
    ax.set_zlabel("Amplitude")
    ax.set_title("Wavefolding Wavetable (sine → folded, decreasing clip amplitude)")

    # Optional: colorbar to get a feel for amplitude distribution
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.show()


# ===================== Audio Rendering ===================== #
def render_audio_from_tables(tables: np.ndarray,
                             sample_rate: int = SAMPLE_RATE,
                             out_file: str = OUT_FILE):
    """
    Flatten the wavetable over time into a 1D audio signal,
    apply a slow amplitude envelope, and write to a mono WAV.
    """
    # Flatten: play table 0, then 1, then 2, ...
    audio = tables.reshape(-1).astype(np.float32)

    # Slow amplitude decay envelope (preserves relative shape over time)
    env = np.linspace(1.0, 0.2, audio.size, dtype=np.float32)
    audio *= env

    # Global normalization to use full int16 range, preserving relative levels
    max_val = np.max(np.abs(audio))
    if max_val < 1e-9:
        print("Audio appears to be all zeros, skipping render.")
        return

    audio /= max_val
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    # Write WAV file
    with wave.open(out_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    duration_sec = audio.size / sample_rate
    print(f"Wrote '{out_file}' ({duration_sec:.2f} s, {sample_rate} Hz, mono)")


# ===================== Main ===================== #
def main():
    tables, phases, thresholds = build_wavetable()
    print(f"Built wavetable: {tables.shape[0]} tables, {tables.shape[1]} samples per table.")
    print(f"Clip threshold goes from {thresholds[0]:.3f} → {thresholds[-1]:.3f}")

    # 3D visualization
    plot_wavetable_3d(tables, phases)

    # Ask if we should generate audio
    ans = input("Generate audio file from this wavetable? [y/N]: ").strip().lower()
    if ans == "y":
        # Optional: choose filename interactively
        custom_name = input(f"Output filename (ENTER for default '{OUT_FILE}'): ").strip()
        out_file = custom_name if custom_name else OUT_FILE

        # Ensure we don't silently overwrite without warning
        if os.path.exists(out_file):
            overwrite = input(f"'{out_file}' exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite != "y":
                print("Not overwriting existing file. Aborting audio render.")
                return

        render_audio_from_tables(tables, out_file=out_file)
    else:
        print("Skipped audio rendering.")


if __name__ == "__main__":
    main()

