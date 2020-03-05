FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /usr/local/bin

RUN pip install --upgrade pip
RUN pip install gym
RUN pip install pandas
RUN pip install ray
RUN pip install ray[tune]
RUN pip install ray[rllib]
RUN pip install procgen

COPY config.py .
COPY transformer.py .
COPY trainer.py .
COPY models_custom.py .

CMD ["python", "trainer.py", "impala"]
