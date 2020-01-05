FROM nvidia/cuda:10.0-devel-ubuntu16.04

ENV USER root

# set working directory
WORKDIR /root

RUN apt-get update

# install basic packages
RUN apt-get install -y openssh-server openssh-client build-essential gfortran wget
RUN apt-get install -y curl git pkg-config zip unzip
RUN apt-get install -y iproute2

# install libcudnn7 (needed for tensorflow)
RUN apt-get install -y libcudnn7=7.6.5.32-1+cuda10.0 libcudnn7-dev=7.6.5.32-1+cuda10.0

# setup python libraries
RUN apt-get install -y python python-pip
RUN apt-get install -y python3 python3-dev python3-pip

RUN pip2 install --upgrade pip
RUN pip2 install supervisor awscli

RUN echo "export PATH=$PATH:/opt/openmpi/bin" >> /etc/bash.bashrc
# libcuda.so.1 is lotated in /usr/local/cuda-10.0/compat, so we need to add it
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi/lib:/usr/local/cuda/include:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-10.0/compat/" >> /etc/bash.bashrc

# install tensorflow
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==1.15.0

# install Open MPI
RUN wget -O /tmp/openmpi.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
WORKDIR /tmp
RUN tar -xvf openmpi.tar.gz
WORKDIR /tmp/openmpi-4.0.0
RUN ./configure --prefix=/opt/openmpi --with-cuda --enable-mpirun-prefix-by-default
RUN make -j $(nproc)
RUN make install

WORKDIR /root

# install gcc and g++ 8
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get install -y g++-8

# install horovod
# TODO: enable pytorch
ENV PATH="${PATH}:/opt/openmpi/bin"
RUN HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip3 install --no-cache-dir horovod

# setup SSH
RUN mkdir -p /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN echo "export VISIBLE=now" >> /etc/profile

RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV SSHDIR /root/.ssh
RUN mkdir -p ${SSHDIR}
RUN touch ${SSHDIR}/sshd_config
RUN ssh-keygen -t rsa -f ${SSHDIR}/ssh_host_rsa_key -N ''
RUN cp ${SSHDIR}/ssh_host_rsa_key.pub ${SSHDIR}/authorized_keys
RUN cp ${SSHDIR}/ssh_host_rsa_key ${SSHDIR}/id_rsa
RUN echo "    IdentityFile ${SSHDIR}/id_rsa" >> /etc/ssh/ssh_config
RUN echo "Host *" >> /etc/ssh/ssh_config && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN chmod -R 600 ${SSHDIR}/* && \
    chown -R ${USER}:${USER} ${SSHDIR}/
# check if ssh agent is running or not, if not, run
RUN eval `ssh-agent -s` && ssh-add ${SSHDIR}/id_rsa

# S3 optimization
RUN aws configure set default.s3.max_concurrent_requests 30
RUN aws configure set default.s3.max_queue_size 10000
RUN aws configure set default.s3.multipart_threshold 64MB
RUN aws configure set default.s3.multipart_chunksize 16MB
RUN aws configure set default.s3.max_bandwidth 4096MB/s
RUN aws configure set default.s3.addressing_style path

RUN git clone https://github.com/aws-samples/deep-learning-models.git /root/deep-learning-models

WORKDIR /

ADD supervisord.conf /etc/supervisor/supervisord.conf
ADD mpi-run.sh supervised-scripts/mpi-run.sh
RUN chmod 755 supervised-scripts/mpi-run.sh

EXPOSE 22

ADD entry-point.sh batch-runtime-scripts/entry-point.sh
RUN chmod 755 batch-runtime-scripts/entry-point.sh

CMD /root/batch-runtime-scripts/entry-point.sh
