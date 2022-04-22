FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DGLBACKEND pytorch
ENV PYTHONPATH "$PYTHONPATH:/spell_check"
WORKDIR /spell_check

RUN apt-get update && apt-get install -y build-essential man

COPY nsc nsc
COPY setup.py .
COPY requirements.txt .
COPY Makefile .
COPY README.rst .
COPY REPRODUCE.rst .

RUN make install

WORKDIR /spell_check/docker
COPY docker/Makefile .
COPY docker/help.sh .

CMD make help && bash
