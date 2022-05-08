FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DGLBACKEND pytorch
ENV PYTHONPATH "$PYTHONPATH:/spell_check"
WORKDIR /spell_check

COPY docker docker
# setup evaluation commands as aliases of the benchmark evaluation script
COPY spell_checking/benchmarks/scripts/evaluate.py .
RUN echo "alias evaluate_tr='python /spell_check/evaluate.py tokenization_repair'" >> ~/.bashrc
RUN echo "alias evaluate_seds='python /spell_check/evaluate.py sed_sequence'" >> ~/.bashrc
RUN echo "alias evaluate_sedw='python /spell_check/evaluate.py sed_words'" >> ~/.bashrc
RUN echo "alias evaluate_sec='python /spell_check/evaluate.py sec'" >> ~/.bashrc

RUN apt-get update && apt-get install -y build-essential curl man

COPY nsc nsc
COPY bin bin
COPY setup.py .
COPY Makefile .
COPY README.rst .
COPY sphinx_docs sphinx_docs

RUN make install

WORKDIR /spell_check/docker
CMD make help && bash
