import numpy as np
from audio import binarize, window_audio

def test_binarize_all_positive():
    vec = np.ones(64, dtype=np.float32)
    h = binarize(vec)
    assert h == (1 << 64) - 1

def test_binarize_all_negative():
    vec = -np.ones(64, dtype=np.float32)
    h = binarize(vec)
    assert h == 0

def test_binarize_mixed():
    vec = np.zeros(64, dtype=np.float32)
    vec[0] = 1.0
    vec[63] = 1.0
    h = binarize(vec)
    assert h == (1 << 0) | (1 << 63)

def test_window_audio_short():
    audio = np.zeros(72000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    assert len(windows) == 0

def test_window_audio_exact():
    audio = np.zeros(120000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    assert len(windows) == 1
    assert windows[0].shape == (120000,)

def test_window_audio_ten_seconds():
    audio = np.zeros(240000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    assert len(windows) == 6
