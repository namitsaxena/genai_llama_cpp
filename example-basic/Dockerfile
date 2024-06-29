FROM python

RUN pip install llama-cpp-python
RUN pip install 'llama-cpp-python[server]'

COPY ./app /app
# COPY ./models/*.gguf /app
COPY ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf /app
COPY ./models/tinyllama-1.1b-chat-v0.3.Q2_K.gguf /app

WORKDIR /app

# CMD ["python", "test.py"]
ENTRYPOINT ["python",  "-m", "llama_cpp.server", "--host", "0.0.0.0", "--model" ]
CMD [ "mistral-7b-instruct-v0.2.Q4_K_M.gguf" ]