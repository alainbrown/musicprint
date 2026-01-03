import torch
import coremltools as ct
from system import MusicPrintSystem
import argparse
import os

def export_to_coreml(args):
    # 1. Load trained model
    print(f"Loading checkpoint: {args.checkpoint_path}")
    # Load onto CPU for export
    system = MusicPrintSystem.load_from_checkpoint(args.checkpoint_path, map_location="cpu")
    model = system.model
    model.eval()

    # 2. Define Dummy Input (5s @ 24kHz)
    example_input = torch.randn(1, 120000)

    # 3. Trace/Script the model
    print("Tracing model...")
    # We use tracing because MERT/Transformers are complex; 
    # tracing captures the actual execution graph.
    traced_model = torch.jit.trace(model, example_input)

    # 4. Convert to CoreML
    print("Converting to CoreML (this may take a few minutes)...")
    # We target the Neural Engine (A15+)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="audio_input", shape=example_input.shape)],
        outputs=[ct.TensorType(name="fingerprint")],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # 5. Add Metadata
    mlmodel.author = "MusicPrint"
    mlmodel.license = "Proprietary"
    mlmodel.short_description = "MERT-v1 based music fingerprinting adapter"
    mlmodel.user_defined_metadata["sample_rate"] = "24000"
    mlmodel.user_defined_metadata["window_seconds"] = "5.0"
    mlmodel.user_defined_metadata["output_dim"] = "64"

    # 6. Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "MusicPrintEncoder.mlpackage")
    mlmodel.save(output_path)
    
    print(f"✅ Successfully exported CoreML model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/app/checkpoints/coreml")
    args = parser.parse_args()
    export_to_coreml(args)
