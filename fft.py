import numpy as np
from scipy import fft
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd
import sys

# Band starting freqs.
mid = 300
hi = 2000

# Import the data and convert to mono if needed.
rate, data = wavfile.read(sys.argv[1])
data = (data / 32768).astype(np.float32).transpose()
if data.ndim == 2:
    data = (data[0] + data[1]) / 2.0
else:
    assert data.ndim == 1

# Split the data into blocks.
blocksize = 4096
r = len(data) % blocksize
if r > 0:
    data = np.append(data, np.zeros(blocksize - r))
blocks = np.array_split(data, len(data) // blocksize)

# FFT each block with windowing.
window = signal.windows.blackmanharris(blocksize, sym=False)
freqs = np.array([fft.rfft(window * b) for b in blocks])

# Convert FFT freqs to bands.
bin_width = rate / 32.0 / 2.0
bin_mid = round(mid / bin_width)
bin_hi = round(hi / bin_width)
bands = np.array(np.array([
    np.sum(np.abs(fs[:bin_mid])),
    np.sum(np.abs(fs[bin_mid:bin_hi])),
    np.sum(np.abs(fs[bin_hi:])),
]) for fs in freqs)

# Play the output of inverse FFT of freqs.
ifs = np.concatenate([fft.irfft(fs) for fs in freqs])
wavfile.write("/tmp/output.wav", rate, ifs)
