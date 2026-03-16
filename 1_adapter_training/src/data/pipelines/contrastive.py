
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dmath
from .ops import audio_reader, audio_decoder

@pipeline_def
def contrastive_pipeline(sample_rate, window_samples, file_root=None, file_list=None, shard_id=0, num_shards=1):
    """
    Dual-view DALI pipeline for self-supervised contrastive learning.
    Outputs: (view_1, view_2, label)
    """
    # 1. Read Signal (Single reader step ensures both views are the same song)
    encoded, label = audio_reader(file_root, file_list, shard_id, num_shards)
    audio = audio_decoder(encoded, sample_rate)
    
    # 2. Setup Global Noise Source
    noise_encoded, _ = fn.readers.file(
        file_root="/vol/data/noise",
        random_shuffle=True,
        name="NoiseReader"
    )
    noise_full = audio_decoder(noise_encoded, sample_rate)

    # Common stats for cropping
    audio_shape = fn.shapes(audio)
    n_samples = fn.slice(audio_shape, 0, 1, axes=[0])
    diff = n_samples - window_samples
    max_anchor = dmath.max(diff, 0)

    def get_augmented_view(seed_val):
        # A. Random Crop (Independent per view)
        anchor = fn.cast(max_anchor * fn.random.uniform(range=[0.0, 1.0]), dtype=types.INT64)
        view = fn.slice(audio, anchor, fn.cast(window_samples, dtype=types.INT64), axes=[0], out_of_bounds_policy="pad")
        
        # B. Independent Noise Slice
        # We use different anchors for noise to ensure views aren't identical
        noise_anchor = fn.cast(fn.random.uniform(range=[0.0, 1000.0]), dtype=types.INT64)
        noise_clip = fn.slice(noise_full, noise_anchor, fn.cast(window_samples, dtype=types.INT64), axes=[0], out_of_bounds_policy="pad")
        
        noise_gain = fn.random.uniform(range=[0.0, 0.3])
        
        # C. Volume Perturbation
        gain = fn.random.uniform(range=[0.5, 1.5])
        
        return (view * gain) + (noise_clip * noise_gain)

    view_1 = get_augmented_view(1)
    view_2 = get_augmented_view(2)

    return view_1, view_2, label
