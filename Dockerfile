FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /spell_check

RUN apt-get update && apt-get install -y build-essential man

COPY nsc .
COPY setup.py .
COPY requirements.txt .
COPY Makefile .
COPY README.rst .
COPY spell_checking .

RUN make install

ENV DGLBACKEND pytorch
ENV PYTHONPATH "$PYTHONPATH:/spell_check"
WORKDIR /spell_check/spell_checking

CMD make help && bash
