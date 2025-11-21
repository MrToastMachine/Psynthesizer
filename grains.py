import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# Load the .wav file
filename = "testSound.wav"  # Replace with your actual file
samplerate, data = wav.read(filename)

# Define the center chunk range
X = len(data) // 2 - 5000  # Example: 5000 samples before the center
Y = len(data) // 2 + 10000  # Example: 5000 samples after the center

# Ensure the range is within valid bounds
X = max(0, X)
Y = min(len(data), Y)

# Extract the chunk
chunk = data[X:Y]

# Play the chunk
sd.play(chunk, samplerate)
sd.wait()  # Wait until audio playback is finished
