from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os

class DALIGPULoader:
    def __init__(self, batch_size, file_root=None, file_list=None, device_id=0, shard_id=0, num_shards=1, sample_rate=24000, window_secs=5.0, augment=False):
        self.batch_size = batch_size
        window_samples = int(window_secs * sample_rate)
        
        self.pipe = audio_pipeline(
            batch_size=batch_size,
            num_threads=4,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            file_root=file_root,
            file_list=file_list, # Pass file_list
            sample_rate=sample_rate,
            window_samples=window_samples,
            augment=augment
        )
        self.pipe.build()

    def __iter__(self):
        return DALIGenericIterator(
            [self.pipe],
            output_map=["audio", "label"],
            reader_name="Reader",
            auto_reset=True
        )

@pipeline_def
def audio_pipeline(sample_rate, window_samples, file_root=None, file_list=None, shard_id=0, num_shards=1, augment=False):
    # 1. Read files
    encoded, label = fn.readers.file(
        file_root=file_root,
        file_list=file_list,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader"
    )
    
    # Ensure label is 64-bit for bitpacked ISRCs
    label = fn.cast(label, dtype=types.INT64)
    
    # 2. Decode & Resample
    # We disable dynamic speed perturbation for stability in V1.
    audio, _ = fn.decoders.audio(
        encoded,
        sample_rate=sample_rate,
        dtype=types.FLOAT,
        downmix=True
    )
    
    # 3. Augmentations (Train Mode Only)
    if augment:
        # A. Add Gaussian Noise
        should_noise = fn.random.coin_flip(probability=0.5)
        noise = fn.random.normal(mean=0.0, stddev=0.01)
        audio = audio + (noise * should_noise)
        
        # B. Volume Perturbation
        gain = fn.random.uniform(range=[0.5, 1.5])
        audio = audio * gain

    # 4. Fixed-length Window
    if augment:
        # Robust Random Crop
        audio_shape = fn.shapes(audio)
        n_samples = fn.slice(audio_shape, 0, 1, axes=[0])
        
        diff = n_samples - window_samples
        max_anchor = fn.math.max(diff, 0)
        
        rand_factor = fn.random.uniform(range=[0.0, 1.0])
        # IMPORTANT: Cast to int64 for slicing
        anchor = fn.cast(max_anchor * rand_factor, dtype=types.INT64)
        
        audio = fn.slice(audio, anchor, window_samples, axes=[0], out_of_bounds_policy="pad")
        
    else:
        # Deterministic Start Crop
        audio = fn.slice(audio, 0, window_samples, axes=[0], out_of_bounds_policy="pad")
    
    return audio, label
