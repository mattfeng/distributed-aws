FROM nvidia/cuda:10.0-devel-ubuntu16.04

ENV USER root

# set working directory
WORKDIR /root

RUN apt-get update

# install basic packages
RUN apt-get install -y openssh-server openssh-client build-essential gfortran wget
RUN apt-get install -y curl git pkg-config zip unzip

# install libcudnn7 (needed for tensorflow)
RUN apt-get install -y libcudnn7=7.6.5.32-1+cuda10.0 libcudnn7-dev=7.6.5.32-1+cuda10.0

# setup python libraries
RUN apt-get install -y python python-pip
RUN apt-get install -y python3 python3-dev python3-pip

RUN pip2 install --upgrade pip
RUN pip2 install supervisor awscli

RUN echo "export PATH=$PATH:/opt/openmpi/bin" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi/lib:/usr/local/cuda/include:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-10.0/compat/" >> /etc/bash.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==1.15.0
