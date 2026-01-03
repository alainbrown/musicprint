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
    
    # 2. Decode & Resample (with Speed Perturbation if augmenting)
    if augment:
        # Speed Perturbation: +/- 5%
        # We achieve this by slightly changing the target sample rate
        # 0.95 * 24000 = 22800, 1.05 * 24000 = 25200
        speed_rate = fn.random.uniform(range=[sample_rate * 0.95, sample_rate * 1.05])
        audio, _ = fn.decoders.audio(
            encoded,
            sample_rate=speed_rate,
            dtype=types.FLOAT,
            downmix=True
        )
    else:
        audio, _ = fn.decoders.audio(
            encoded,
            sample_rate=sample_rate,
            dtype=types.FLOAT,
            downmix=True
        )
    
    # 3. Augmentations (Train Mode Only)
    if augment:
        # A. Random Speed/Pitch Shift (+/- 5%)
        # DALI doesn't have a direct 'pitch_shift' operator for raw audio in the graph easily
        # without external libraries, but we can simulate speed change via resampling
        # or use a random time stretch if available.
        # For V1, we will stick to simpler additive noise which is natively supported.
        
        # B. Add Gaussian Noise (Simulate sensor noise)
        # 50% chance to apply noise
        should_noise = fn.random.coin_flip(probability=0.5)
        noise = fn.random.normal(mean=0.0, stddev=0.01) # Low level noise
        audio = audio + (noise * should_noise)
        
        # C. Volume Perturbation (0.5x to 1.5x)
        gain = fn.random.uniform(range=[0.5, 1.5])
        audio = audio * gain

    # 4. Fixed-length Window
    if augment:
        # Robust Random Crop
        # 1. Get length of the audio
        audio_shape = fn.shapes(audio)
        # 2. Extract time dimension (dim 0)
        # DALI shapes returns [d0, d1, ...], we need just d0 as a scalar-like
        n_samples = fn.slice(audio_shape, 0, 1, axes=[0])
        
        # 3. Calculate range: max(0, length - window)
        # Note: DALI math ops work on tensors
        diff = n_samples - window_samples
        max_anchor = fn.math.max(diff, 0)
        
        # 4. Generate random float 0-1 and scale to range
        rand_factor = fn.random.uniform(range=[0.0, 1.0])
        anchor = max_anchor * rand_factor
        
        # 5. Slice with padding for safety
        # We cast anchor to int via slice args usually, but DALI needs specific types
        # fn.slice can take tensor anchors.
        audio = fn.slice(audio, anchor, window_samples, axes=[0], out_of_bounds_policy="pad")
        
    else:
        # Deterministic Start Crop
        audio = fn.slice(audio, 0, window_samples, axes=[0], out_of_bounds_policy="pad")
    
    return audio, label
