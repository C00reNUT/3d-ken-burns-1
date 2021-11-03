FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENTRYPOINT []
COPY ./src /work/src
COPY ./requirement /work/requirement.txt

WORKDIR /work/src
RUN ./scripts/install.sh