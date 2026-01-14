
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def audio_reader(file_root=None, file_list=None, shard_id=0, num_shards=1, random_shuffle=True):
    """Standard DALI file reader for audio files"""
    encoded, label = fn.readers.file(
        file_root=file_root,
        file_list=file_list,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader"
    )
    # Ensure label is 64-bit for bitpacked ISRCs
    label = fn.cast(label, dtype=types.INT64)
    return encoded, label

def audio_decoder(encoded, sample_rate=24000):
    """Standard DALI audio decoder and resampler"""
    audio, _ = fn.decoders.audio(
        encoded,
        sample_rate=sample_rate,
        dtype=types.FLOAT,
        downmix=True
    )
    return audio

def noise_reader(file_root="/vol/data/noise"):
    """Reads environmental noise from the flat directory structure"""
    noise_encoded, _ = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        name="NoiseReader"
    )
    return noise_encoded
