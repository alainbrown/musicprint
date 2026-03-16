
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from .ops import audio_reader, audio_decoder

@pipeline_def
def inference_pipeline(sample_rate, window_samples, file_root=None, file_list=None, shard_id=0, num_shards=1):
    """
    Single-view DALI pipeline for deterministic inference/validation.
    Outputs: (audio, label)
    """
    # 1. Read and Decode Signal
    encoded, label = audio_reader(file_root, file_list, shard_id, num_shards, random_shuffle=False)
    audio = audio_decoder(encoded, sample_rate)
    
    # 2. Deterministic Start Crop
    audio_crop = fn.slice(
        audio, 
        fn.cast(0, dtype=types.INT64), 
        fn.cast(window_samples, dtype=types.INT64), 
        axes=[0], 
        out_of_bounds_policy="pad"
    )
    
    return audio_crop, label
