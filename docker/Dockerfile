# Starting from here in case I want to run a jupyter-lab instance for notebooks along the way
FROM jupyter/minimal-notebook:ubuntu-18.04

USER root

# Declare volumes for input and output
# /data contains datasets and the like
# /pickles contains pickles saved as we process a job
VOLUME /data
VOLUME /pickles

# Install all of the necessary packages
RUN sudo apt-get -y update && apt-get install -y \
    vim \
    git \
    python3-pip \
    cmake \
    curl \
    bzip2 \
    libsndfile1 \
    libportaudio2

RUN pip3 install 'numpy==1.20' \
    tensorflow \
    librosa \
    sounddevice \
    google-cloud-speech \
    'vosk==0.3.27' \
    pandas \
    'scikit-learn==0.23.1' \
    sklearn \
    deap \
    playsound \
    'h5py==2.10.0'

# Load all of the necessary pieces from this distribution into the Docker image
# Note that the data directory is not added...I map that as /data
# Note that the pickles directory is not added...I map that as /pickles

RUN mkdir /workspace

# See up-front materials. Assumes that certain downloads for 3rd-party datasets and VOSK model files have occured
COPY . /workspace/

WORKDIR /workspace
RUN ln -s /pickles pickles
RUN ln -s /data data

CMD ["bash"]
