
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types

class DALILoader:
    """
    Generic DALI Loader Factory.
    Connects a pipeline definition to a PyTorch-compatible iterator.
    """
    def __init__(self, pipeline_fn, output_map, batch_size, **kwargs):
        self.batch_size = batch_size
        self.output_map = output_map
        
        # Build the graph
        self.pipe = pipeline_fn(
            batch_size=batch_size,
            num_threads=4, # Ideal for mid-range CPU
            **kwargs
        )
        self.pipe.build()

    def __iter__(self):
        return DALIGenericIterator(
            [self.pipe],
            output_map=self.output_map,
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP
        )

    def __len__(self):
        # DALI iterators don't strictly have a length, but we can query the reader
        return self.pipe.epoch_size("Reader") // self.batch_size
