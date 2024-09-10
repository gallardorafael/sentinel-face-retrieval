FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . ./sentinel-workspace
WORKDIR /sentinel-workspace

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "face-retrieval.py", "--server.port=8501", "--server.address=0.0.0.0"]