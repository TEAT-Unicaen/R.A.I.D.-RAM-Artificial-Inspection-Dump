FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

RUN pip3 install \
    cryptography \
    opencv-python \
    numpy \
    matplotlib \
    aiohttp \
    asyncio \
    tqdm 

WORKDIR /workspace