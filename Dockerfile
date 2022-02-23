FROM continuumio/anaconda3:2021.05

RUN conda create -n spell_check python=3.8

WORKDIR /spell_check
ENV PYTHONPATH "$PYTHONPATH:/spell_check"

COPY . .

RUN conda develop .

RUN apt update && \
    apt install -y wget git gcc g++ patch make cmake build-essential graphviz graphviz-dev libhunspell-dev libaspell-dev

RUN pip install -r requirements.txt && \
    pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

RUN cd third_party && git clone --recursive git@github.com:WojciechMula/aspell-python.git && \
    cd aspell-python && python setup.py install
RUN cd third_party/errant && python setup.py install
RUN cd third_party && wget https://languagetool.org/download/ngram-data/ngrams-en-20150817.zip -P languagetool && \
    unzip languagetool/ngrams-en-20150817.zip -d languagetool && rm languagetool/ngrams-en-20150817.zip

RUN python -m spacy download en_core_web_trf

WORKDIR /spell_check/spelling_correction

CMD /spell_check/spelling_correction/welcome.sh

# BUILD
# -----------------------------------------------------------
# docker build -t gnn_spelling_correction .

# RUN
# (Make sure you have docker version >= 19.03, a nvidia driver installed
# and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
# if you want to run the container with GPU support)
# -----------------------------------------------------------
# docker run -it [--gpus all] -p <local_port>:8501 \
# -v /nfs/students/sebastian-walter/masters_thesis/data:/masters_thesis/data \
# -v /nfs/students/sebastian-walter/masters_thesis/experiments:/masters_thesis/experiments \
# gnn_spelling_correction
#
# where
#  - <local_port> is the port you want to view the demo on

