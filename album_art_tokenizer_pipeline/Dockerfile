# Use NVIDIA's optimized PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install dependencies for Image Processing, VQ-VAE, and CoreML export
RUN pip install --no-cache-dir \
    torchvision \
    pillow \
    tqdm \
    pandas \
    coremltools \
    wandb \
    scipy \
    matplotlib

WORKDIR /app

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose Jupyter port if needed for notebooks
EXPOSE 8888

CMD ["/bin/bash"]
