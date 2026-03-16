import struct
import numpy as np

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000   # 5 seconds at 24kHz
STRIDE_SAMPLES = 24000    # 1 second stride


def binarize(embedding):
    """Convert a 64-dim float embedding to a uint64 hash via sign bits.

    Matches the C++ and index pipeline convention:
    bit[i] = 1 if embedding[i] > 0, packed little-endian.
    """
    bits = (embedding > 0).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return struct.unpack("<Q", packed.tobytes())[0]


def window_audio(audio, sr=SAMPLE_RATE):
    """Split audio into 5-second windows with 1-second stride.

    Args:
        audio: 1D numpy array of float32 samples
        sr: sample rate (must be 24000)

    Returns:
        List of 1D numpy arrays, each 120000 samples.
    """
    if len(audio) < WINDOW_SAMPLES:
        return []
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= len(audio):
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES
    return windows


def load_and_resample(audio_path):
    """Load audio from any format, resample to 24kHz mono.

    Uses pydub+ffmpeg for format support.
    Returns a 1D numpy float32 array normalized to [-1, 1].
    """
    from pydub import AudioSegment

    seg = AudioSegment.from_file(audio_path)
    seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= 32768.0
    return samples
