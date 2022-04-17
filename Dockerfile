FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /spell_check
ENV PYTHONPATH "$PYTHONPATH:/spell_check"

RUN apt-get update && apt-get install -y build-essential

COPY . .

RUN make install
RUN python -m spacy download en_core_web_lg

WORKDIR /spell_check/spell_checking

CMD /spell_check/spell_checking/welcome.sh
