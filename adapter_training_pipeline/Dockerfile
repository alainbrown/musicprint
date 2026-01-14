# Use NVIDIA's optimized PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies for audio processing
# libavcodec-extra contains the codecs that DALI's decoders.audio often needs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Install transformers (Hugging Face) and JupyterLab and Lightning
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    jupyterlab \
    ipywidgets \
    pytorch-lightning \
    coremltools \
    wandb \
    faiss-gpu \
    scipy \
    librosa \
    datasets

WORKDIR /app

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose Jupyter port
EXPOSE 8888

# Launch bash by default if no command provided
CMD ["/bin/bash"]