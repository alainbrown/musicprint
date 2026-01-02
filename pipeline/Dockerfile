# Use NVIDIA's optimized PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install transformers (Hugging Face)
RUN pip install --no-cache-dir transformers

WORKDIR /app

# Set python path to include src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["/bin/bash"]
