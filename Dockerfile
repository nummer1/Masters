FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/local/bin

RUN pip install gym
RUN pip install pandas
RUN pip install ray
RUN pip install procgen==0.9.4

COPY config.py .
COPY transformer.py .
COPY trainer.py .
COPY models_custom.py .

CMD ["python", "trainer.py", "impala"]
