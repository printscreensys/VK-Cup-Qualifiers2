FROM continuumio/anaconda3:latest

CMD mkdir /opt/results
WORKDIR /opt/results
RUN pip install catboost
COPY model.cbm model.cbm
COPY users.tsv users.tsv
COPY baseline_always_zero.py baseline_always_zero.py

CMD python baseline_always_zero.py /tmp/data/test.tsv > /opt/results/result.tsv
