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
    # Type Masquerading: Temporarily substitute the dynamic class with a static
    # local definition to generate a clean type signature in the TorchScript schema.
    # This resolves parser errors caused by long, auto-generated remote code paths.
    
    # Recursive patcher
    patched_modules = []
    
    def recursive_patch(module, prefix=""):
        # Check if the module's class is dynamic (from transformers_modules)
        if module.__class__.__module__.startswith("transformers_modules"):
            original_class = module.__class__
            clean_name = original_class.__name__
            
            # Create a local class inheriting from the original
            # We use type() to create the class dynamically with the clean name
            # This ensures TorchScript sees "clean_name" (or similar) instead of the long path
            class PatchedClass(original_class):
                pass
            
            # Rename the class to avoid "PatchedClass" showing up everywhere if we want
            # though TorchScript might use the local variable name. 
            # Let's just use the class structure.
            PatchedClass.__name__ = clean_name
            PatchedClass.__qualname__ = clean_name
            
            print(f"Patching {prefix} ({original_class.__name__}) -> {clean_name}")
            module.__class__ = PatchedClass
            patched_modules.append((module, original_class))
            
        # Recurse
        for name, child in module.named_children():
            recursive_patch(child, prefix=f"{prefix}.{name}")

    try:
        recursive_patch(model.backbone.backbone, prefix="backbone")
        traced_model = torch.jit.trace(model, example_input)
    finally:
        # Restore all classes
        print(f"Restoring {len(patched_modules)} patched modules...")
        for module, original_class in reversed(patched_modules):
            module.__class__ = original_class

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
    
    # Save TorchScript for the Indexer
    pt_path = os.path.join(args.output_dir, "encoder.pt")
    traced_model.save(pt_path)
    print(f"✅ Saved TorchScript model to: {pt_path}")

    # Save CoreML for the App
    output_path = os.path.join(args.output_dir, "MusicPrintEncoder.mlpackage")
    mlmodel.save(output_path)
    
    print(f"✅ Successfully exported CoreML model to: {output_path}")

def export(args):
    """
    Main export entry point.
    args: Namespace or object with attributes:
        checkpoint_path, output_dir
    """
    export_to_coreml(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/vol/release")
    args = parser.parse_args()
    export(args)
