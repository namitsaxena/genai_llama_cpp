FROM python

RUN pip install llama-cpp-python
RUN pip install Flask

ARG MODEL_DIR=models
ARG MODEL_NAME=tinyllama-1.1b-chat-v0.3.Q2_K.gguf

COPY ./app /app
COPY ${MODEL_DIR}/${MODEL_NAME} /app
WORKDIR /app

CMD ["python", "server.py"]