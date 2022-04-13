FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /spell_check
ENV PYTHONPATH "$PYTHONPATH:/spell_check"

COPY . .

RUN pip install -r requirements.txt && \
    pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

RUN python -m spacy download en_core_web_lg

WORKDIR /spell_check/spelling_correction

CMD /spell_check/spelling_correction/welcome.sh

# BUILD
# -----------------------------------------------------------
# docker build -t gsc .

# RUN
# (Make sure you have docker version >= 19.03, a nvidia driver installed
# and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
# if you want to run the container with GPU support)
# -----------------------------------------------------------
# docker run -it [--gpus all] gsc
#
