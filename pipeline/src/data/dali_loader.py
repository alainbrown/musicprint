from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os

@pipeline_def
def audio_pipeline(file_root, sample_rate, window_samples):
    # 1. Read files (CPU stage of DALI)
    encoded, label = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        name="Reader"
    )
    
    # 2. Decode & Resample (GPU)
    # This happens on the GPU if the device is set to 'mixed' or if we use specific GPU operators
    audio, _ = fn.decoders.audio(
        encoded,
        sample_rate=sample_rate,
        dtype=types.FLOAT,
        downmix=True
    )
    
    # 3. Fixed-length Window (GPU)
    # We ensure every batch has the exact same number of samples for the transformer
    audio = fn.pad(audio, fill_value=0, axes=[0], shape=[window_samples])
    audio = fn.slice(audio, 0, window_samples, axes=[0])
    
    return audio, label

class DALIGPULoader:
    def __init__(self, file_root, batch_size, device_id=0, sample_rate=24000, window_secs=5.0):
        self.batch_size = batch_size
        window_samples = int(window_secs * sample_rate)
        
        self.pipe = audio_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=device_id,
            file_root=file_root,
            sample_rate=sample_rate,
            window_samples=window_samples
        )
        self.pipe.build()

    def __iter__(self):
        # Maps DALI outputs to PyTorch tensors
        return DALIGenericIterator(
            [self.pipe],
            output_map=["audio", "label"],
            reader_name="Reader",
            auto_reset=True
        )
