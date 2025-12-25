#FROM nvidia/cuda:10.2-base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

########## DOCKER SETUP START ##########
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH=/opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./requirements.yaml /tmp/requirements.yaml
RUN conda update conda
RUN conda env create --name evaluation -f /tmp/requirements.yaml
# Reinstall PyTorch with CUDA support via pip (conda sometimes installs CPU version)
# PyTorch 1.13.1 was built with CUDA 11.7, which is compatible with CUDA 11.8
RUN /opt/conda/envs/evaluation/bin/pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN echo "conda activate evaluation" >> ~/.bashrc

# setup permissions for non-root user to access necessary files and directories
RUN chown -R 1000:1000 /opt/conda/pkgs
RUN useradd someuser -u 1000 --create-home
RUN mkdir -p /opt/conda/envs/evaluation/lib/python3.7/site-packages/data && chown -R 1000:1000 /opt/conda/envs/evaluation/lib/python3.7/site-packages/data
USER someuser

# setup conda env variables
ENV PATH=/opt/conda/envs/evaluation/bin:$PATH
ENV CONDA_DEFAULT_ENV=evaluation
# Use conda's libstdc++ to avoid GLIBCXX version issues
ENV LD_LIBRARY_PATH=/opt/conda/envs/evaluation/lib:$LD_LIBRARY_PATH

# install evaluation specific code
RUN mkdir /home/someuser/data
RUN (echo 'from torchvision.datasets import CIFAR10'; echo '_ = CIFAR10(root="/home/someuser/data", download=True)') | python
# CIFAR10 will be downloaded automatically at runtime if not present

# run evaluation
COPY ./evaluate.sh .
COPY ./verify_gpu.py /home/someuser
COPY ./Towards-Robust-Neural-Networks-via-Close-loop-Control /home/someuser/
COPY ./evaluate.py /home/someuser
COPY ./model_client.py /home/someuser
COPY ./model_server.py /home/someuser
COPY ./utils.py /home/someuser
ENTRYPOINT ["bash", "evaluate.sh"]
